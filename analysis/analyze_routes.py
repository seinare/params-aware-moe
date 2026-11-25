#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import train_gpt


@dataclass
class ModelSpec:
    name: str
    checkpoint: Path
    router_type: str
    num_experts: int
    top_k: int = 16
    product_dim: int = 128
    router_m: int = 4
    aux_coef: float = 0.0


def infer_spec(path: Path) -> ModelSpec | None:
    name = path.stem
    if "linear" in name:
        match = re.search(r"linear_e(\d+)k(\d+)", name)
        if not match:
            return None
        return ModelSpec(
            name=name,
            checkpoint=path,
            router_type="linear",
            num_experts=int(match.group(1)),
            top_k=int(match.group(2)),
            product_dim=256 if int(match.group(1)) >= 128 else 128,
            aux_coef=0.01,
        )
    if "pa_" in name or "pa_e" in name:
        match = re.search(r"pa_e(\d+)k(\d+).*?pd(\d+)", name)
        if not match:
            return None
        return ModelSpec(
            name=name,
            checkpoint=path,
            router_type="param",
            num_experts=int(match.group(1)),
            top_k=int(match.group(2)),
            product_dim=int(match.group(3)),
            router_m=4,
            aux_coef=0.0 if "lb0" in name else 0.01,
        )
    return None


def load_tokens(data_path: Path, num_tokens: int, seq_len: int) -> torch.Tensor:
    stream = train_gpt.TokenStream(str(data_path / "fineweb_train_*.bin"))
    usable = (num_tokens // seq_len) * seq_len
    return stream.take(usable + 1).to(dtype=torch.long)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if "__quant_format__" in obj:
            return train_gpt.dequantize_state_dict_int8(obj)
    return obj


def build_model(spec: ModelSpec, device: torch.device) -> train_gpt.GPT:
    return train_gpt.GPT(
        vocab_size=1024,
        num_layers=8,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=4,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10_000.0,
        qk_gain_init=0.3,
        use_param_aware_moe=True,
        use_expert_parallel=False,
        num_experts=spec.num_experts,
        moe_top_k=spec.top_k,
        expert_inner_dim=64,
        mlp_activation="relu_square",
        router_m=spec.router_m,
        router_product_dim=spec.product_dim,
        share_router_key_proj=True,
        num_mlp_pools=8,
        router_key_detach=True,
        router_aux_loss_coef=spec.aux_coef,
        route_metrics=False,
        use_dense_mlp_residual=False,
        dense_mlp_residual_mult=1,
        moe_router_type=spec.router_type,
    ).to(device)


def state_load(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    obj = torch.load(checkpoint, map_location="cpu")
    state = unwrap_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    serious_missing = [k for k in missing if not k.endswith(("route_counts", "route_prob_sums", "route_gate_entropy_sum", "route_token_count"))]
    if serious_missing or unexpected:
        raise RuntimeError(
            f"load_state_dict mismatch for {checkpoint}: missing={serious_missing[:8]} unexpected={unexpected[:8]}"
        )
    model.to(device)


def empty_stats(num_layers: int, num_experts: int, device: torch.device):
    return {
        "gate_sum": torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64),
        "prob_sum": torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64),
        "count": torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64),
        "tokens": torch.zeros(num_layers, device=device, dtype=torch.float64),
    }


def record_router(module, x, layer_idx: int, stats):
    bsz, seqlen, dim = x.shape
    if isinstance(module, train_gpt.ParamAwareMoE):
        q = module.query(x).view(bsz, seqlen, module.router_m, module.router_product_dim)
        q = F.rms_norm(q, (q.size(-1),))
        expert_key_act = module.pool.make_keys(module.key_embed, module.router_key_detach)
        if module.share_key_proj:
            keys = torch.einsum("emh,hp->emp", expert_key_act, module.key_proj).to(dtype=q.dtype)
        else:
            keys = torch.einsum("emh,ehp->emp", expert_key_act, module.key_proj).to(dtype=q.dtype)
        keys = F.rms_norm(keys, (keys.size(-1),))
        scores = torch.einsum("btmp,emp->btme", q, keys) * module.score_scale
        probs = F.softmax(scores.float(), dim=-1).mean(dim=2)
    elif isinstance(module, train_gpt.LinearRouterMoE):
        probs = F.softmax(module.router(x).float(), dim=-1)
    else:
        raise TypeError(type(module))

    top_probs, top_idx = torch.topk(probs, k=module.top_k, dim=-1)
    gates = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    flat_idx = top_idx.reshape(-1)
    stats["prob_sum"][layer_idx].scatter_add_(0, torch.arange(probs.size(-1), device=x.device), probs.sum(dim=(0, 1)).double())
    stats["count"][layer_idx].scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float64))
    stats["gate_sum"][layer_idx].scatter_add_(0, flat_idx, gates.reshape(-1).double())
    stats["tokens"][layer_idx] += float(bsz * seqlen)
    return top_idx, gates


def forward_collect(model: train_gpt.GPT, input_ids: torch.Tensor, stats) -> None:
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips = []
    sparse_layer = 0

    for i in range(model.num_encoder_layers):
        block = model.blocks[i]
        mix = block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + block.attn_scale.to(dtype=x.dtype)[None, None, :] * block.attn(block.attn_norm(x))
        mlp_in = block.mlp_norm(x)
        top_idx, gates = record_router(block.mlp, mlp_in, sparse_layer, stats)
        sparse_layer += 1
        mlp_out = block.mlp.pool.selected_forward(
            mlp_in.reshape(-1, mlp_in.size(-1)),
            top_idx.reshape(-1, block.mlp.top_k),
            gates.reshape(-1, block.mlp.top_k),
        ).reshape_as(x)
        x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        skips.append(x)

    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        block = model.blocks[model.num_encoder_layers + i]
        mix = block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + block.attn_scale.to(dtype=x.dtype)[None, None, :] * block.attn(block.attn_norm(x))
        mlp_in = block.mlp_norm(x)
        top_idx, gates = record_router(block.mlp, mlp_in, sparse_layer, stats)
        sparse_layer += 1
        mlp_out = block.mlp.pool.selected_forward(
            mlp_in.reshape(-1, mlp_in.size(-1)),
            top_idx.reshape(-1, block.mlp.top_k),
            gates.reshape(-1, block.mlp.top_k),
        ).reshape_as(x)
        x = x + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out


def summarize(spec: ModelSpec, stats, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_mean = stats["gate_sum"] / stats["tokens"].clamp_min(1.0)[:, None]
    prob_mean = stats["prob_sum"] / stats["tokens"].clamp_min(1.0)[:, None]
    select_rate = stats["count"] / stats["tokens"].clamp_min(1.0)[:, None]
    sorted_gate, sorted_idx = torch.sort(gate_mean, dim=-1, descending=True)

    torch.save(
        {
            "name": spec.name,
            "gate_mean": gate_mean.cpu().float(),
            "prob_mean": prob_mean.cpu().float(),
            "select_rate": select_rate.cpu().float(),
            "sorted_gate_mean": sorted_gate.cpu().float(),
            "sorted_expert_idx": sorted_idx.cpu().int(),
            "tokens_per_layer": stats["tokens"].cpu().long(),
        },
        out_dir / f"{spec.name}.pt",
    )

    with (out_dir / f"{spec.name}_layer_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "layer",
                "experts",
                "tokens",
                "gate_entropy_norm",
                "prob_entropy_norm",
                "gate_max_over_mean",
                "select_max_over_mean",
                "dead_select_frac",
                "top1_gate_mean",
                "top16_gate_mass",
                "top32_gate_mass",
            ],
        )
        writer.writeheader()
        rows = []
        for layer in range(gate_mean.size(0)):
            gm = gate_mean[layer]
            pm = prob_mean[layer]
            sr = select_rate[layer]
            sg = sorted_gate[layer]
            gate_entropy = -(gm * gm.clamp_min(1e-12).log()).sum()
            prob_entropy = -(pm * pm.clamp_min(1e-12).log()).sum()
            denom = math.log(gm.numel())
            row = {
                "model": spec.name,
                "layer": layer,
                "experts": gm.numel(),
                "tokens": int(stats["tokens"][layer].item()),
                "gate_entropy_norm": float((gate_entropy / denom).item()),
                "prob_entropy_norm": float((prob_entropy / denom).item()),
                "gate_max_over_mean": float((gm.max() / gm.mean().clamp_min(1e-12)).item()),
                "select_max_over_mean": float((sr.max() / sr.mean().clamp_min(1e-12)).item()),
                "dead_select_frac": float((sr == 0).float().mean().item()),
                "top1_gate_mean": float(sg[0].item()),
                "top16_gate_mass": float(sg[: min(16, sg.numel())].sum().item()),
                "top32_gate_mass": float(sg[: min(32, sg.numel())].sum().item()),
            }
            rows.append(row)
            writer.writerow(row)
    return {
        "model": spec.name,
        "layers": gate_mean.size(0),
        "experts": gate_mean.size(1),
        "tokens": int(stats["tokens"][0].item()),
        "gate_entropy_norm_mean": sum(r["gate_entropy_norm"] for r in rows) / len(rows),
        "gate_max_over_mean_mean": sum(r["gate_max_over_mean"] for r in rows) / len(rows),
        "select_max_over_mean_mean": sum(r["select_max_over_mean"] for r in rows) / len(rows),
        "top16_gate_mass_mean": sum(r["top16_gate_mass"] for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("route_analysis"))
    parser.add_argument("--tokens", type=int, default=2_097_152)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-seqs", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpts = sorted(args.checkpoint_dir.rglob("*.pt"))
    specs = [s for p in ckpts if (s := infer_spec(p)) is not None and p.name != "latest.pt"]
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if any(w in s.name for w in wanted)]
    if not specs:
        raise SystemExit(f"No router checkpoints found under {args.checkpoint_dir}")

    tokens = load_tokens(args.data_path, args.tokens, args.seq_len)
    num_seqs = (tokens.numel() - 1) // args.seq_len
    summary = []
    for spec in specs:
        print(f"=== {spec.name} router={spec.router_type} E={spec.num_experts} K={spec.top_k} P={spec.product_dim}", flush=True)
        model = build_model(spec, device)
        state_load(model, spec.checkpoint, device)
        model.eval()
        stats = empty_stats(8, spec.num_experts, device)
        with torch.no_grad():
            for start in range(0, num_seqs, args.batch_seqs):
                end = min(start + args.batch_seqs, num_seqs)
                local = tokens[start * args.seq_len : end * args.seq_len + 1].to(device, non_blocking=True)
                x = local[:-1].reshape(-1, args.seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    forward_collect(model, x, stats)
                if (end * args.seq_len) % 262144 == 0 or end == num_seqs:
                    print(f"{spec.name}: {end * args.seq_len}/{num_seqs * args.seq_len} tokens", flush=True)
        summary.append(summarize(spec, stats, args.out_dir))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
