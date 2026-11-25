"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import contextlib
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as dist_fc
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class StepProfiler:
    OUTER_SECTION_NAMES = {"data", "forward", "backward", "grad_sync", "grad_clip", "optimizer"}

    def __init__(self, enabled: bool, device: torch.device, inner_sections: bool = True):
        self.enabled = enabled and device.type == "cuda"
        self.device = device
        self.inner_sections = inner_sections
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    @contextlib.contextmanager
    def section(self, name: str):
        if not self.enabled or (
            not self.inner_sections
            and name not in self.OUTER_SECTION_NAMES
            and not name.startswith("optimizer_")
        ):
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.setdefault(name, []).append((start, end))

    def collect(self) -> dict[str, float]:
        if not self.enabled or not self.events:
            self.events.clear()
            return {}
        torch.cuda.synchronize(self.device)
        out = {
            f"time/profile/{name}_ms": sum(start.elapsed_time(end) for start, end in events)
            for name, events in self.events.items()
        }
        self.events.clear()
        return out


ACTIVE_PROFILER: StepProfiler | None = None


@contextlib.contextmanager
def profile_section(name: str):
    if ACTIVE_PROFILER is None:
        yield
    else:
        with ACTIVE_PROFILER.section(name):
            yield


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def autograd_all_gather_cat(x: Tensor, dim: int = 0) -> Tensor:
    if not distributed_active():
        return x
    if x.is_cuda:
        return dist_fc.all_gather_tensor_autograd(x, gather_dim=dim, group=dist.group.WORLD)
    return torch.cat(tuple(dist_nn.all_gather(x)), dim=dim)


def autograd_all_reduce_sum(x: Tensor) -> Tensor:
    if not distributed_active():
        return x
    return dist_nn.all_reduce(x, op=dist.ReduceOp.SUM)


class AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        output_split_sizes: tuple[int, ...],
        input_split_sizes: tuple[int, ...],
    ) -> Tensor:
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        out_shape = (sum(output_split_sizes), *x.shape[1:])
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        dist.all_to_all_single(
            out,
            x.contiguous(),
            output_split_sizes=list(output_split_sizes),
            input_split_sizes=list(input_split_sizes),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grad_x = AllToAllSingle.apply(grad_out.contiguous(), ctx.input_split_sizes, ctx.output_split_sizes)
        return grad_x, None, None


def autograd_all_to_all_single(
    x: Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> Tensor:
    return AllToAllSingle.apply(x, tuple(output_split_sizes), tuple(input_split_sizes))


def exchange_counts(send_counts: Tensor) -> Tensor:
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous())
    return recv_counts

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    profile_step_time = bool(int(os.environ.get("PROFILE_STEP_TIME", "0")))
    profile_inner_sections = bool(int(os.environ.get("PROFILE_INNER_SECTIONS", "1")))
    route_metrics = bool(int(os.environ.get("ROUTE_METRICS", "1")))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    use_param_aware_moe = bool(int(os.environ.get("USE_PARAM_AWARE_MOE", "0")))
    use_expert_parallel = bool(int(os.environ.get("USE_EXPERT_PARALLEL", "0")))
    moe_router_type = os.environ.get("MOE_ROUTER_TYPE", "param").lower().replace("-", "_")
    num_experts = int(os.environ.get("NUM_EXPERTS", 1024))
    moe_top_k = int(os.environ.get("MOE_TOP_K", 32))
    expert_inner_dim = int(os.environ.get("EXPERT_INNER_DIM", 64))
    mlp_activation = os.environ.get("MLP_ACTIVATION", "relu_square").lower().replace("-", "_")
    router_m = int(os.environ.get("ROUTER_M", 1))
    router_product_dim = int(os.environ.get("ROUTER_PRODUCT_DIM", 128))
    share_router_key_proj = bool(int(os.environ.get("SHARE_ROUTER_KEY_PROJ", "0")))
    num_mlp_pools = int(os.environ.get("NUM_MLP_POOLS", 1))
    router_key_detach = bool(int(os.environ.get("ROUTER_KEY_DETACH", "0")))
    router_aux_loss_coef = float(os.environ.get("ROUTER_AUX_LOSS_COEF", 0.0))
    use_dense_mlp_residual = bool(int(os.environ.get("USE_DENSE_MLP_RESIDUAL", "0")))
    dense_mlp_residual_mult = int(os.environ.get("DENSE_MLP_RESIDUAL_MULT", 1))
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "1")))
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "1")))
    skip_final_eval = bool(int(os.environ.get("SKIP_FINAL_EVAL", "0")))
    skip_post_training_eval = bool(int(os.environ.get("SKIP_POST_TRAINING_EVAL", "0")))
    final_full_val = bool(int(os.environ.get("FINAL_FULL_VAL", "0")))
    save_checkpoint = bool(int(os.environ.get("SAVE_CHECKPOINT", "0")))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoint")
    wandb_project = os.environ.get("WANDB_PROJECT", "")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", run_id)

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_3d_impl = os.environ.get("MUON_3D_IMPL", "batched").lower().replace("-", "_")
    muon_reduce_updates = bool(int(os.environ.get("MUON_REDUCE_UPDATES", "1")))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def zeropower_via_newtonschulz5_batched(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Batched expert-wise orthogonalization for [E, rows, cols] gradients.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm(dim=(1, 2), keepdim=True) + eps
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.transpose(1, 2) if transposed else X


def muonize_grad(g: Tensor, steps: int) -> Tensor:
    if g.ndim == 2:
        out = zeropower_via_newtonschulz5(g, steps=steps)
        return out * max(1, g.size(0) / g.size(1)) ** 0.5
    if g.ndim == 3:
        if os.environ.get("MUON_3D_IMPL", "batched").lower().replace("-", "_") == "batched":
            out = zeropower_via_newtonschulz5_batched(g, steps=steps)
            return out * max(1, g.size(1) / g.size(2)) ** 0.5
        chunks = []
        scale = max(1, g.size(1) / g.size(2)) ** 0.5
        for expert_grad in g:
            chunks.append(zeropower_via_newtonschulz5(expert_grad, steps=steps) * scale)
        return torch.stack(chunks, dim=0)
    raise ValueError(f"Muon only supports 2D/3D params, got shape {tuple(g.shape)}")


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        reduce_updates: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                reduce_updates=reduce_updates,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            reduce_updates = group["reduce_updates"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                owns_update = (i % world_size == rank) if reduce_updates else True
                if owns_update and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = muonize_grad(g, steps=backend_steps)
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed and reduce_updates:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if args.val_max_tokens > 0:
        total_seqs = min(total_seqs, max(args.val_max_tokens // args.train_seq_len, 1))
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def check_mlp_activation(activation: str) -> str:
    activation = activation.lower().replace("-", "_")
    if activation not in {"relu_square", "swiglu"}:
        raise ValueError(f"MLP_ACTIVATION must be relu_square or swiglu, got {activation!r}")
    return activation


if triton is not None:
    @triton.jit
    def _moe_token_sum_combine_kernel(
        expert_out,
        gates,
        inverse_order,
        out,
        num_tokens: tl.constexpr,
        top_k: tl.constexpr,
        dim: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        block_id = tl.program_id(1)
        cols = block_id * block_d + tl.arange(0, block_d)
        mask = cols < dim
        acc = tl.zeros((block_d,), tl.float32)
        for k in range(top_k):
            original_pos = token_id * top_k + k
            sorted_pos = tl.load(inverse_order + original_pos)
            gate = tl.load(gates + sorted_pos).to(tl.float32)
            vals = tl.load(expert_out + sorted_pos * dim + cols, mask=mask, other=0.0).to(tl.float32)
            acc += gate * vals
        tl.store(out + token_id * dim + cols, acc, mask=mask)

    @triton.jit
    def _moe_token_sum_combine_bwd_kernel(
        grad_out,
        expert_out,
        gates,
        inverse_order,
        grad_expert_out,
        grad_gates,
        num_tokens: tl.constexpr,
        top_k: tl.constexpr,
        dim: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        block_id = tl.program_id(1)
        cols = block_id * block_d + tl.arange(0, block_d)
        mask = cols < dim
        grad_vals = tl.load(grad_out + token_id * dim + cols, mask=mask, other=0.0).to(tl.float32)
        for k in range(top_k):
            original_pos = token_id * top_k + k
            sorted_pos = tl.load(inverse_order + original_pos)
            gate = tl.load(gates + sorted_pos).to(tl.float32)
            expert_vals = tl.load(expert_out + sorted_pos * dim + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(grad_expert_out + sorted_pos * dim + cols, grad_vals * gate, mask=mask)
            grad_gate_part = tl.sum(grad_vals * expert_vals, axis=0)
            tl.atomic_add(grad_gates + sorted_pos, grad_gate_part, sem="relaxed")


class _TritonTokenSumCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_out: Tensor, sorted_gates: Tensor, sorted_tokens: Tensor, order: Tensor, num_tokens: int):
        if triton is None:
            raise RuntimeError("MOE_COMBINE_IMPL=triton requires triton to be installed")
        if not expert_out.is_cuda:
            raise RuntimeError("MOE_COMBINE_IMPL=triton requires CUDA tensors")
        num_assignments, dim = expert_out.shape
        top_k = num_assignments // num_tokens
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(num_assignments, device=order.device, dtype=order.dtype)
        out = torch.empty((num_tokens, dim), device=expert_out.device, dtype=expert_out.dtype)
        block_d = 128
        grid = (num_tokens, triton.cdiv(dim, block_d))
        _moe_token_sum_combine_kernel[grid](
            expert_out,
            sorted_gates,
            inverse_order,
            out,
            num_tokens,
            top_k,
            dim,
            block_d,
        )
        ctx.save_for_backward(expert_out, sorted_gates, inverse_order)
        ctx.num_tokens = num_tokens
        ctx.top_k = top_k
        ctx.dim = dim
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        expert_out, sorted_gates, inverse_order = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_expert_out = torch.empty_like(expert_out)
        grad_gates = torch.zeros(sorted_gates.shape, device=sorted_gates.device, dtype=torch.float32)
        grid = (ctx.num_tokens, triton.cdiv(ctx.dim, ctx.block_d))
        _moe_token_sum_combine_bwd_kernel[grid](
            grad_out,
            expert_out,
            sorted_gates,
            inverse_order,
            grad_expert_out,
            grad_gates,
            ctx.num_tokens,
            ctx.top_k,
            ctx.dim,
            ctx.block_d,
        )
        return grad_expert_out, grad_gates.to(dtype=sorted_gates.dtype), None, None, None


def moe_combine(
    expert_out: Tensor,
    sorted_gates: Tensor,
    sorted_tokens: Tensor,
    order: Tensor,
    num_tokens: int,
    out_dtype: torch.dtype,
) -> Tensor:
    impl = os.environ.get("MOE_COMBINE_IMPL", "index_add").lower().replace("-", "_")
    if impl == "triton":
        return _TritonTokenSumCombine.apply(expert_out, sorted_gates, sorted_tokens, order, num_tokens).to(dtype=out_dtype)
    if impl in {"token_sum", "contiguous_reduce"}:
        combined = torch.empty_like(expert_out)
        combined[order] = sorted_gates[:, None].to(dtype=expert_out.dtype) * expert_out
        return combined.view(num_tokens, -1, expert_out.size(-1)).sum(dim=1).to(dtype=out_dtype)
    if impl != "index_add":
        raise ValueError(f"MOE_COMBINE_IMPL must be index_add, token_sum, or triton, got {impl!r}")
    out = torch.zeros((num_tokens, expert_out.size(-1)), device=expert_out.device, dtype=out_dtype)
    out.index_add_(0, sorted_tokens, (sorted_gates[:, None].to(dtype=expert_out.dtype) * expert_out).to(dtype=out_dtype))
    return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, activation: str):
        super().__init__()
        self.activation = check_mlp_activation(activation)
        hidden = mlp_mult * dim
        if self.activation == "swiglu":
            self.up = CastedLinear(dim, 2 * hidden, bias=False)
        else:
            self.up = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        up = self.up(x)
        if self.activation == "swiglu":
            gate, up = up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        else:
            hidden = F.relu(up).square()
        return self.proj(hidden)


class FineGrainedMLPPool(nn.Module):
    def __init__(
        self,
        num_experts: int,
        dim: int,
        inner_dim: int,
        activation: str,
        expert_parallel_rank: int = 0,
        expert_parallel_world_size: int = 1,
    ):
        super().__init__()
        if num_experts % expert_parallel_world_size != 0:
            raise ValueError(
                f"NUM_EXPERTS={num_experts} must divide expert_parallel_world_size={expert_parallel_world_size}"
            )
        self.activation = check_mlp_activation(activation)
        self.num_experts = num_experts
        self.expert_parallel_rank = expert_parallel_rank
        self.expert_parallel_world_size = expert_parallel_world_size
        self.local_num_experts = num_experts // expert_parallel_world_size
        self.local_expert_start = expert_parallel_rank * self.local_num_experts
        self.local_expert_end = self.local_expert_start + self.local_num_experts
        self.dim = dim
        self.inner_dim = inner_dim
        up_rows = 2 * inner_dim if self.activation == "swiglu" else inner_dim
        self.up_weight = nn.Parameter(torch.empty(self.local_num_experts, up_rows, dim))
        self.down_weight = nn.Parameter(torch.empty(self.local_num_experts, dim, inner_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_weight)

    def activate(self, up: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gate, up = up.chunk(2, dim=-1)
            return F.silu(gate) * up
        return F.relu(up).square()

    def make_keys(self, key_embed: Tensor, detach: bool) -> Tensor:
        up_weight = self.up_weight.detach() if detach else self.up_weight
        return self.activate(torch.einsum("emd,eod->emo", key_embed, up_weight))

    def selected_forward(self, x_flat: Tensor, expert_idx: Tensor, gates: Tensor) -> Tensor:
        if self.expert_parallel_world_size > 1:
            return self.selected_forward_all_to_all(x_flat, expert_idx, gates)
        if x_flat.is_cuda and hasattr(F, "grouped_mm"):
            return self.selected_forward_grouped_mm(x_flat, expert_idx, gates)
        return self.selected_forward_loop(x_flat, expert_idx, gates)

    def local_expert_forward(self, x_sorted: Tensor, sorted_experts: Tensor) -> Tensor:
        counts = torch.bincount(sorted_experts, minlength=self.local_num_experts)
        offsets = torch.cumsum(counts, dim=0).to(dtype=torch.int32)
        if x_sorted.is_cuda and hasattr(F, "grouped_mm"):
            up_weight = self.up_weight.transpose(1, 2).contiguous()
            down_weight = self.down_weight.transpose(1, 2).contiguous()
            x_compute = x_sorted.to(dtype=up_weight.dtype)
            hidden = self.activate(F.grouped_mm(x_compute, up_weight, offs=offsets)).to(dtype=down_weight.dtype)
            return F.grouped_mm(hidden, down_weight, offs=offsets).to(dtype=x_sorted.dtype)

        chunks: list[Tensor] = []
        start = 0
        for local_expert, count in enumerate(counts.tolist()):
            end = start + int(count)
            if end > start:
                x_chunk = x_sorted[start:end]
                hidden = self.activate(F.linear(x_chunk.to(dtype=self.up_weight.dtype), self.up_weight[local_expert]))
                chunks.append(F.linear(hidden, self.down_weight[local_expert]).to(dtype=x_chunk.dtype))
            start = end
        return torch.cat(chunks, dim=0) if chunks else x_sorted.new_empty((0, self.dim))

    @torch.compiler.disable
    def selected_forward_all_to_all(self, x_flat: Tensor, expert_idx: Tensor, gates: Tensor) -> Tensor:
        num_tokens = x_flat.size(0)
        top_k = expert_idx.size(1)
        flat_experts = expert_idx.reshape(-1)
        flat_gates = gates.reshape(-1)
        flat_tokens = torch.arange(num_tokens, device=x_flat.device, dtype=torch.long).repeat_interleave(top_k)

        with profile_section("moe_dispatch_pack"):
            owner = torch.div(flat_experts, self.local_num_experts, rounding_mode="floor")
            local_experts = flat_experts - owner * self.local_num_experts
            _, send_order = torch.sort(owner)
            owner_sorted = owner[send_order]
            token_send = flat_tokens[send_order]
            expert_send = local_experts[send_order]
            gate_send = flat_gates[send_order]

            # Send each token hidden state at most once to each remote owner. The per-expert
            # assignments are still sent separately as compact metadata and gate vectors.
            owner_token_key = owner * num_tokens + flat_tokens
            unique_keys, unique_inverse = torch.unique(owner_token_key, sorted=True, return_inverse=True)
            x_owner_sorted = torch.div(unique_keys, num_tokens, rounding_mode="floor")
            x_token_send = unique_keys - x_owner_sorted * num_tokens
            x_send = x_flat[x_token_send]

            meta_send_counts = torch.bincount(owner_sorted, minlength=self.expert_parallel_world_size).to(torch.long)
            x_send_counts = torch.bincount(x_owner_sorted, minlength=self.expert_parallel_world_size).to(torch.long)
            meta_recv_counts = exchange_counts(meta_send_counts)
            x_recv_counts = exchange_counts(x_send_counts)
            meta_send_splits = [int(x) for x in meta_send_counts.cpu().tolist()]
            meta_recv_splits = [int(x) for x in meta_recv_counts.cpu().tolist()]
            x_send_splits = [int(x) for x in x_send_counts.cpu().tolist()]
            x_recv_splits = [int(x) for x in x_recv_counts.cpu().tolist()]

            x_owner_offsets = torch.cumsum(x_send_counts, dim=0) - x_send_counts
            x_slot_send = unique_inverse[send_order] - x_owner_offsets[owner_sorted]
            meta_send = torch.stack((token_send, expert_send, x_slot_send), dim=1).contiguous()
            meta_recv = torch.empty((int(meta_recv_counts.sum().item()), 3), device=x_flat.device, dtype=torch.long)

        with profile_section("moe_ep_dispatch_a2a"):
            x_recv = autograd_all_to_all_single(x_send, x_recv_splits, x_send_splits)
            gate_recv = autograd_all_to_all_single(gate_send[:, None], meta_recv_splits, meta_send_splits).squeeze(1)
            dist.all_to_all_single(
                meta_recv,
                meta_send,
                output_split_sizes=meta_recv_splits,
                input_split_sizes=meta_send_splits,
            )

        with profile_section("moe_grouped_gemm"):
            recv_experts = meta_recv[:, 1]
            if recv_experts.numel() == 0:
                expert_out = x_recv.new_empty((0, self.dim))
            else:
                meta_source = torch.repeat_interleave(
                    torch.arange(self.expert_parallel_world_size, device=x_flat.device, dtype=torch.long),
                    torch.as_tensor(meta_recv_splits, device=x_flat.device, dtype=torch.long),
                )
                x_recv_offsets = torch.cumsum(x_recv_counts, dim=0) - x_recv_counts
                sorted_experts, compute_order = torch.sort(recv_experts)
                x_sorted = x_recv[x_recv_offsets[meta_source[compute_order]] + meta_recv[compute_order, 2]]
                gates_sorted = gate_recv[compute_order]
                expert_out_sorted = self.local_expert_forward(x_sorted, sorted_experts)
                expert_out_sorted = gates_sorted[:, None].to(dtype=expert_out_sorted.dtype) * expert_out_sorted
                expert_out = torch.empty_like(expert_out_sorted)
                expert_out[compute_order] = expert_out_sorted

        with profile_section("moe_ep_combine_a2a"):
            returned = autograd_all_to_all_single(expert_out, meta_send_splits, meta_recv_splits)

        with profile_section("moe_dispatch_scatter"):
            out = torch.zeros_like(x_flat)
            out.index_add_(0, token_send, returned)
        return out

    def selected_forward_grouped_mm(self, x_flat: Tensor, expert_idx: Tensor, gates: Tensor) -> Tensor:
        num_tokens = x_flat.size(0)
        top_k = expert_idx.size(1)
        flat_experts = expert_idx.reshape(-1)
        flat_gates = gates.reshape(-1)
        flat_tokens = torch.arange(num_tokens, device=x_flat.device).repeat_interleave(top_k)
        flat_experts = flat_experts - self.local_expert_start

        with profile_section("moe_dispatch_pack"):
            sorted_experts, order = torch.sort(flat_experts)
            sorted_tokens = flat_tokens[order]
            sorted_gates = flat_gates[order]
            x_sorted = x_flat[sorted_tokens]

            counts = torch.bincount(sorted_experts, minlength=self.local_num_experts)
            offsets = torch.cumsum(counts, dim=0).to(dtype=torch.int32)
            up_weight = self.up_weight.transpose(1, 2).contiguous()
            down_weight = self.down_weight.transpose(1, 2).contiguous()

        with profile_section("moe_grouped_gemm"):
            x_compute = x_sorted.to(dtype=up_weight.dtype)
            hidden = self.activate(F.grouped_mm(x_compute, up_weight, offs=offsets)).to(dtype=down_weight.dtype)
            expert_out = F.grouped_mm(hidden, down_weight, offs=offsets)

        with profile_section("moe_dispatch_scatter"):
            return moe_combine(expert_out, sorted_gates, sorted_tokens, order, num_tokens, x_flat.dtype)

    def selected_forward_loop(self, x_flat: Tensor, expert_idx: Tensor, gates: Tensor) -> Tensor:
        out = torch.zeros_like(x_flat)
        flat_experts = expert_idx.reshape(-1)
        flat_gates = gates.reshape(-1)
        token_idx = torch.arange(x_flat.size(0), device=x_flat.device).repeat_interleave(expert_idx.size(1))
        for local_expert, expert in enumerate(range(self.local_expert_start, self.local_expert_end)):
            selected = flat_experts == expert
            if not bool(selected.any()):
                continue
            selected_tokens = token_idx[selected]
            x_selected = x_flat[selected_tokens]
            hidden = self.activate(F.linear(x_selected.to(dtype=self.up_weight.dtype), self.up_weight[local_expert]))
            expert_out = F.linear(hidden, self.down_weight[local_expert]).to(dtype=out.dtype)
            expert_out = flat_gates[selected, None].to(dtype=expert_out.dtype) * expert_out
            out.index_add_(0, selected_tokens, expert_out)
        return out


class ParamAwareMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        pool: FineGrainedMLPPool,
        top_k: int,
        router_m: int,
        router_product_dim: int,
        share_key_proj: bool,
        router_key_detach: bool,
        aux_loss_coef: float,
        route_metrics: bool,
        use_dense_residual: bool,
        dense_residual_mult: int,
        mlp_activation: str,
    ):
        super().__init__()
        if top_k <= 0 or top_k > pool.num_experts:
            raise ValueError(f"MOE_TOP_K must be in [1, NUM_EXPERTS], got {top_k}")
        if router_m <= 0:
            raise ValueError(f"ROUTER_M must be positive, got {router_m}")
        if router_product_dim <= 0:
            raise ValueError(f"ROUTER_PRODUCT_DIM must be positive, got {router_product_dim}")
        self.pool = pool
        self.top_k = top_k
        self.router_m = router_m
        self.router_product_dim = router_product_dim
        self.share_key_proj = share_key_proj
        self.router_key_detach = router_key_detach
        self.aux_loss_coef = aux_loss_coef
        self.route_metrics_enabled = route_metrics
        self.score_scale = 1.0 / math.sqrt(router_product_dim)
        self.query = CastedLinear(dim, router_m * router_product_dim, bias=False)
        self.key_embed = nn.Parameter(torch.empty(pool.local_num_experts, router_m, dim))
        key_proj_shape = (
            (pool.inner_dim, router_product_dim)
            if share_key_proj
            else (pool.local_num_experts, pool.inner_dim, router_product_dim)
        )
        self.key_proj = nn.Parameter(torch.empty(*key_proj_shape))
        self.dense_residual = MLP(dim, dense_residual_mult, mlp_activation) if use_dense_residual else None
        self.register_buffer("route_counts", torch.zeros(pool.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("route_prob_sums", torch.zeros(pool.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("route_gate_entropy_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("route_token_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.lb_prob_sum: Tensor | None = None
        self.lb_counts: Tensor | None = None
        self.lb_num_assignments = 0
        self.lb_num_tokens = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.key_embed, mean=0.0, std=1.0 / math.sqrt(self.key_embed.size(-1)))
        nn.init.kaiming_uniform_(self.key_proj, a=math.sqrt(5))

    def load_balance_loss(self, probs: Tensor, top_idx: Tensor) -> Tensor:
        # Switch-style auxiliary loss. probs is dense softmax mass; selected traffic is counted sparsely.
        mean_prob = probs.float().mean(dim=(0, 1))
        flat_idx = top_idx.flatten()
        counts = torch.zeros(probs.size(-1), device=probs.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        mean_selected = counts / float(top_idx.numel())
        return probs.size(-1) * torch.sum(mean_prob * mean_selected)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        expert_parallel = self.pool.expert_parallel_world_size > 1
        x_flat = x.reshape(-1, dim)
        bsz_route, seqlen_route = bsz, seqlen
        x_route = x

        with profile_section("router_query"):
            q = self.query(x_route).view(bsz_route, seqlen_route, self.router_m, self.router_product_dim)
            q = F.rms_norm(q, (q.size(-1),))
        with profile_section("router_key"):
            expert_key_act = self.pool.make_keys(self.key_embed, self.router_key_detach)
            if self.share_key_proj:
                local_keys = torch.einsum("emh,hp->emp", expert_key_act, self.key_proj).to(dtype=q.dtype)
            else:
                local_keys = torch.einsum("emh,ehp->emp", expert_key_act, self.key_proj).to(dtype=q.dtype)
            keys = autograd_all_gather_cat(local_keys, dim=0) if expert_parallel else local_keys
            keys = F.rms_norm(keys, (keys.size(-1),))
        with profile_section("router_score"):
            scores = torch.einsum("btmp,emp->btme", q, keys) * self.score_scale
        with profile_section("router_softmax_topk"):
            probs = F.softmax(scores.float(), dim=-1).mean(dim=2).to(dtype=x.dtype)
            top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            gates = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            flat_idx = top_idx.flatten()
            counts = torch.zeros(self.pool.num_experts, device=probs.device, dtype=torch.float32)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        if self.training:
            self.lb_prob_sum = probs.float().sum(dim=(0, 1))
            self.lb_counts = counts
            self.lb_num_assignments = int(top_idx.numel())
            self.lb_num_tokens = int(bsz_route * seqlen_route)
        if self.training and self.route_metrics_enabled:
            with torch.no_grad():
                self.route_counts.add_(counts.detach())
                self.route_prob_sums.add_(probs.float().sum(dim=(0, 1)))
                gate_entropy = -(gates.float() * gates.float().clamp_min(1e-12).log()).sum(dim=-1)
                self.route_gate_entropy_sum.add_(gate_entropy.sum())
                self.route_token_count.add_(float(bsz_route * seqlen_route))

        with profile_section("moe_dispatch_total"):
            out = self.pool.selected_forward(x_flat, top_idx.reshape(-1, self.top_k), gates.reshape(-1, self.top_k))
        out = out.reshape(bsz, seqlen, dim)
        if self.dense_residual is not None:
            with profile_section("dense_residual"):
                out = out + self.dense_residual(x)

        return out, x.new_zeros(())


class LinearRouterMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        pool: FineGrainedMLPPool,
        top_k: int,
        aux_loss_coef: float,
        route_metrics: bool,
        use_dense_residual: bool,
        dense_residual_mult: int,
        mlp_activation: str,
    ):
        super().__init__()
        if top_k <= 0 or top_k > pool.num_experts:
            raise ValueError(f"MOE_TOP_K must be in [1, NUM_EXPERTS], got {top_k}")
        self.pool = pool
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.route_metrics_enabled = route_metrics
        self.router = CastedLinear(dim, pool.num_experts, bias=False)
        self.dense_residual = MLP(dim, dense_residual_mult, mlp_activation) if use_dense_residual else None
        self.register_buffer("route_counts", torch.zeros(pool.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("route_prob_sums", torch.zeros(pool.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("route_gate_entropy_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("route_token_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.lb_prob_sum: Tensor | None = None
        self.lb_counts: Tensor | None = None
        self.lb_num_assignments = 0
        self.lb_num_tokens = 0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        x_flat = x.reshape(-1, dim)
        with profile_section("router_score"):
            scores = self.router(x)
        with profile_section("router_softmax_topk"):
            probs = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            gates = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            flat_idx = top_idx.flatten()
            counts = torch.zeros(self.pool.num_experts, device=probs.device, dtype=torch.float32)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        if self.training:
            self.lb_prob_sum = probs.float().sum(dim=(0, 1))
            self.lb_counts = counts
            self.lb_num_assignments = int(top_idx.numel())
            self.lb_num_tokens = int(bsz * seqlen)
        if self.training and self.route_metrics_enabled:
            with torch.no_grad():
                self.route_counts.add_(counts.detach())
                self.route_prob_sums.add_(probs.float().sum(dim=(0, 1)))
                gate_entropy = -(gates.float() * gates.float().clamp_min(1e-12).log()).sum(dim=-1)
                self.route_gate_entropy_sum.add_(gate_entropy.sum())
                self.route_token_count.add_(float(bsz * seqlen))

        with profile_section("moe_dispatch_total"):
            out = self.pool.selected_forward(x_flat, top_idx.reshape(-1, self.top_k), gates.reshape(-1, self.top_k))
        out = out.reshape(bsz, seqlen, dim)
        if self.dense_residual is not None:
            with profile_section("dense_residual"):
                out = out + self.dense_residual(x)
        return out, x.new_zeros(())


SPARSE_MOE_TYPES = (ParamAwareMoE, LinearRouterMoE)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        moe_pool: FineGrainedMLPPool | None = None,
        moe_top_k: int = 0,
        mlp_activation: str = "relu_square",
        router_m: int = 1,
        router_product_dim: int = 128,
        share_router_key_proj: bool = False,
        router_key_detach: bool = False,
        router_aux_loss_coef: float = 0.0,
        route_metrics: bool = True,
        use_dense_mlp_residual: bool = False,
        dense_mlp_residual_mult: int = 1,
        moe_router_type: str = "param",
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        if moe_pool is None:
            self.mlp = MLP(dim, mlp_mult, mlp_activation)
        elif moe_router_type == "param":
            self.mlp = ParamAwareMoE(
                dim,
                moe_pool,
                moe_top_k,
                router_m,
                router_product_dim,
                share_router_key_proj,
                router_key_detach,
                router_aux_loss_coef,
                route_metrics,
                use_dense_mlp_residual,
                dense_mlp_residual_mult,
                mlp_activation,
            )
        elif moe_router_type == "linear":
            self.mlp = LinearRouterMoE(
                dim,
                moe_pool,
                moe_top_k,
                router_aux_loss_coef,
                route_metrics,
                use_dense_mlp_residual,
                dense_mlp_residual_mult,
                mlp_activation,
            )
        else:
            raise ValueError(f"MOE_ROUTER_TYPE must be param or linear, got {moe_router_type!r}")
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        aux_loss = x.new_zeros(())
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x, aux_loss


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_param_aware_moe: bool = False,
        use_expert_parallel: bool = False,
        num_experts: int = 1024,
        moe_top_k: int = 32,
        expert_inner_dim: int = 64,
        mlp_activation: str = "relu_square",
        router_m: int = 1,
        router_product_dim: int = 128,
        share_router_key_proj: bool = False,
        num_mlp_pools: int = 1,
        router_key_detach: bool = False,
        router_aux_loss_coef: float = 0.0,
        route_metrics: bool = True,
        use_dense_mlp_residual: bool = False,
        dense_mlp_residual_mult: int = 1,
        moe_router_type: str = "param",
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        expert_parallel_world_size = dist.get_world_size() if use_expert_parallel and distributed_active() else 1
        expert_parallel_rank = dist.get_rank() if use_expert_parallel and distributed_active() else 0
        if use_expert_parallel and not use_param_aware_moe:
            raise ValueError("USE_EXPERT_PARALLEL=1 requires USE_PARAM_AWARE_MOE=1")
        if use_expert_parallel and expert_parallel_world_size <= 1:
            raise ValueError("USE_EXPERT_PARALLEL=1 requires distributed launch with WORLD_SIZE > 1")
        if use_param_aware_moe and num_mlp_pools <= 0:
            raise ValueError(f"NUM_MLP_POOLS must be positive when MoE is enabled, got {num_mlp_pools}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.mlp_pools = nn.ModuleList(
            [
                FineGrainedMLPPool(
                    num_experts,
                    model_dim,
                    expert_inner_dim,
                    mlp_activation,
                    expert_parallel_rank=expert_parallel_rank,
                    expert_parallel_world_size=expert_parallel_world_size,
                )
                for _ in range(num_mlp_pools)
            ]
            if use_param_aware_moe
            else []
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    self.mlp_pools[i % num_mlp_pools] if use_param_aware_moe else None,
                    moe_top_k,
                    mlp_activation,
                    router_m,
                    router_product_dim,
                    share_router_key_proj,
                    router_key_detach,
                    router_aux_loss_coef,
                    route_metrics,
                    use_dense_mlp_residual,
                    dense_mlp_residual_mult,
                    moe_router_type,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.last_ce_loss: Tensor | None = None
        self.last_aux_loss: Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def router_metrics(self, reset: bool = True) -> dict[str, float]:
        counts: list[Tensor] = []
        prob_sums: list[Tensor] = []
        gate_entropy_sum = 0.0
        token_count = 0.0
        for module in self.modules():
            if not isinstance(module, SPARSE_MOE_TYPES):
                continue
            counts.append(module.route_counts.detach().float().cpu().clone())
            prob_sums.append(module.route_prob_sums.detach().float().cpu().clone())
            gate_entropy_sum += float(module.route_gate_entropy_sum.detach().cpu())
            token_count += float(module.route_token_count.detach().cpu())
            if reset:
                module.route_counts.zero_()
                module.route_prob_sums.zero_()
                module.route_gate_entropy_sum.zero_()
                module.route_token_count.zero_()
        if not counts or token_count <= 0:
            return {}
        counts_t = torch.stack(counts)
        prob_t = torch.stack(prob_sums)
        load = counts_t / counts_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        prob = prob_t / prob_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        load_entropy = -(load * load.clamp_min(1e-12).log()).sum(dim=-1)
        prob_entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        max_load = load.max(dim=-1).values
        mean_load = load.mean(dim=-1)
        dead_frac = (counts_t == 0).float().mean(dim=-1)
        num_experts = counts_t.size(-1)
        norm = math.log(num_experts)
        return {
            "router/load_entropy": float(load_entropy.mean().item()),
            "router/load_entropy_norm": float((load_entropy / norm).mean().item()) if norm > 0 else 0.0,
            "router/prob_entropy": float(prob_entropy.mean().item()),
            "router/prob_entropy_norm": float((prob_entropy / norm).mean().item()) if norm > 0 else 0.0,
            "router/max_load": float(max_load.mean().item()),
            "router/mean_load": float(mean_load.mean().item()),
            "router/max_over_mean_load": float((max_load / mean_load.clamp_min(1e-12)).mean().item()),
            "router/dead_expert_frac": float(dead_frac.mean().item()),
            "router/gate_entropy": gate_entropy_sum / token_count,
        }

    def pool_load_balance_loss(self, dtype: torch.dtype) -> Tensor:
        pool_stats: dict[int, dict[str, object]] = {}
        for module in self.modules():
            if not isinstance(module, SPARSE_MOE_TYPES):
                continue
            if module.aux_loss_coef <= 0.0 or module.lb_prob_sum is None or module.lb_counts is None:
                continue
            key = id(module.pool)
            if key not in pool_stats:
                pool_stats[key] = {
                    "num_experts": module.pool.num_experts,
                    "coef": module.aux_loss_coef,
                    "prob_sum": module.lb_prob_sum,
                    "counts": module.lb_counts,
                    "tokens": module.lb_num_tokens,
                    "assignments": module.lb_num_assignments,
                }
            else:
                stats = pool_stats[key]
                stats["prob_sum"] = stats["prob_sum"] + module.lb_prob_sum
                stats["counts"] = stats["counts"] + module.lb_counts
                stats["tokens"] = int(stats["tokens"]) + module.lb_num_tokens
                stats["assignments"] = int(stats["assignments"]) + module.lb_num_assignments

        aux_loss: Tensor | None = None
        for stats in pool_stats.values():
            tokens = max(int(stats["tokens"]), 1)
            assignments = max(int(stats["assignments"]), 1)
            mean_prob = stats["prob_sum"] / float(tokens)
            mean_selected = stats["counts"] / float(assignments)
            pool_loss = float(stats["coef"]) * int(stats["num_experts"]) * torch.sum(mean_prob * mean_selected)
            aux_loss = pool_loss if aux_loss is None else aux_loss + pool_loss
        if aux_loss is None:
            return torch.zeros((), dtype=dtype, device=self.tok_emb.weight.device)
        return aux_loss.to(dtype=dtype)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x, block_aux = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, block_aux = self.blocks[self.num_encoder_layers + i](x, x0)
        aux_loss = self.pool_load_balance_loss(x.dtype)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        self.last_ce_loss = ce_loss.detach()
        self.last_aux_loss = aux_loss.detach()
        return ce_loss + aux_loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global ACTIVE_PROFILER, zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_batched

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.moe_router_type not in {"param", "linear"}:
        raise ValueError(f"MOE_ROUTER_TYPE must be param or linear, got {args.moe_router_type!r}")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    zeropower_via_newtonschulz5_batched = torch.compile(zeropower_via_newtonschulz5_batched)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    if args.use_expert_parallel:
        if not distributed:
            raise ValueError("USE_EXPERT_PARALLEL=1 requires torchrun/distributed launch")
        if not args.use_param_aware_moe:
            raise ValueError("USE_EXPERT_PARALLEL=1 requires USE_PARAM_AWARE_MOE=1")
        if args.num_experts % world_size != 0:
            raise ValueError(f"NUM_EXPERTS={args.num_experts} must divide WORLD_SIZE={world_size}")

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    wandb_run = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
        if args.wandb_project:
            try:
                import wandb

                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config={k: v for k, v in vars(args.__class__).items() if not k.startswith("_")},
                )
            except Exception as exc:
                print(f"wandb:init_failed:{type(exc).__name__}:{exc}")

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_param_aware_moe=args.use_param_aware_moe,
        use_expert_parallel=args.use_expert_parallel,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        expert_inner_dim=args.expert_inner_dim,
        mlp_activation=args.mlp_activation,
        router_m=args.router_m,
        router_product_dim=args.router_product_dim,
        share_router_key_proj=args.share_router_key_proj,
        num_mlp_pools=args.num_mlp_pools,
        router_key_detach=args.router_key_detach,
        router_aux_loss_coef=args.router_aux_loss_coef,
        route_metrics=args.route_metrics,
        use_dense_mlp_residual=args.use_dense_mlp_residual,
        dense_mlp_residual_mult=args.dense_mlp_residual_mult,
        moe_router_type=args.moe_router_type,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=args.compile_fullgraph)
        if args.compile_model
        else base_model
    )
    model: nn.Module = (
        compiled_model
        if args.use_expert_parallel
        else DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    )

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix and 3D expert tensors in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars and router key embeddings use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    def is_local_expert_param(name: str) -> bool:
        return args.use_expert_parallel and (
            "key_embed" in name or "pool.up_weight" in name or "pool.down_weight" in name
        )

    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim in {2, 3}
        and "key_embed" not in name
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and not is_local_expert_param(name)
    ]
    local_expert_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim in {2, 3}
        and "key_embed" not in name
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and is_local_expert_param(name)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim not in {2, 3}
        or "key_embed" in name
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        if not is_local_expert_param(name)
    ]
    local_expert_scalar_params = [
        p
        for name, p in block_named_params
        if (
            p.ndim not in {2, 3}
            or "key_embed" in name
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        )
        and is_local_expert_param(name)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        reduce_updates=args.muon_reduce_updates,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if local_expert_matrix_params:
        optimizer_expert_muon = Muon(
            local_expert_matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            reduce_updates=False,
        )
        for group in optimizer_expert_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_expert_muon)
    if local_expert_scalar_params:
        optimizer_expert_scalar = torch.optim.Adam(
            [{"params": local_expert_scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_expert_scalar)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"compile_model:{args.compile_model} compile_fullgraph:{args.compile_fullgraph} "
        f"route_metrics:{args.route_metrics}"
    )
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    if args.use_param_aware_moe:
        log0(
            f"param_aware_moe:enabled router_type:{args.moe_router_type} experts:{args.num_experts} top_k:{args.moe_top_k} "
            f"expert_inner_dim:{args.expert_inner_dim} activation:{args.mlp_activation} router_m:{args.router_m} "
            f"product_dim:{args.router_product_dim} share_key_proj:{args.share_router_key_proj} "
            f"mlp_pools:{args.num_mlp_pools} key_detach:{args.router_key_detach} "
            f"aux_coef:{args.router_aux_loss_coef} dense_residual:{args.use_dense_mlp_residual} "
            f"expert_parallel:{args.use_expert_parallel}"
        )
    else:
        log0(f"param_aware_moe:disabled mlp_mult:{args.mlp_mult} activation:{args.mlp_activation}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(f"muon_config:backend_steps:{args.muon_backend_steps} 3d_impl:{args.muon_3d_impl} reduce_updates:{args.muon_reduce_updates}")
    log0(
        f"optimizer_params:muon:{sum(p.numel() for p in matrix_params)} "
        f"scalar_adam:{sum(p.numel() for p in scalar_params)} "
        f"local_expert_muon:{sum(p.numel() for p in local_expert_matrix_params)} "
        f"local_expert_adam:{sum(p.numel() for p in local_expert_scalar_params)}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"warmdown_iters:{args.warmdown_iters} max_wallclock_seconds:{args.max_wallclock_seconds:.3f} "
        f"profile_step_time:{args.profile_step_time} profile_inner_sections:{args.profile_inner_sections} "
        f"final_full_val:{args.final_full_val} "
        f"save_checkpoint:{args.save_checkpoint} checkpoint_dir:{args.checkpoint_dir}"
    )
    log0(f"seed:{args.seed}")
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model_params": n_params,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
                "token_lr": token_lr,
            },
            allow_val_change=True,
        )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    replicated_grad_sync_params = [base_model.tok_emb.weight, *matrix_params, *scalar_params]
    if base_model.lm_head is not None:
        replicated_grad_sync_params.append(base_model.lm_head.weight)

    def sync_replicated_grads() -> None:
        if not args.use_expert_parallel:
            return
        for p in replicated_grad_sync_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def save_checkpoint(step: int, final_val_loss: float | None = None, final_val_bpb: float | None = None) -> None:
        if not args.save_checkpoint:
            return
        if distributed and not args.use_expert_parallel and not master_process:
            return
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = args.wandb_run_name or args.run_id
        rank_suffix = f"_rank{rank:03d}" if args.use_expert_parallel else ""
        payload = {
            "step": step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
            "config": {k: v for k, v in vars(args.__class__).items() if not k.startswith("_")},
            "final_val_loss": final_val_loss,
            "final_val_bpb": final_val_bpb,
            "rank": rank,
            "world_size": world_size,
            "expert_parallel": args.use_expert_parallel,
        }
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_step{step:06d}{rank_suffix}.pt"
        torch.save(payload, checkpoint_path)
        latest_path = checkpoint_dir / (f"latest_rank{rank:03d}.pt" if args.use_expert_parallel else "latest.pt")
        torch.save(payload, latest_path)
        log0(f"checkpoint_saved:{checkpoint_path} bytes:{checkpoint_path.stat().st_size}")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed and not args.use_expert_parallel:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed and not args.use_expert_parallel:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    final_checkpoint_val_loss: float | None = None
    final_checkpoint_val_bpb: float | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = (last_step and not args.skip_final_eval) or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            original_val_max_tokens = args.val_max_tokens
            if last_step and args.final_full_val:
                args.val_max_tokens = 0
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            args.val_max_tokens = original_val_max_tokens
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/bpb": val_bpb,
                        "time/train_ms": training_time_ms,
                        "time/step_avg_ms": training_time_ms / max(step, 1),
                    },
                    step=step,
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if last_step and args.final_full_val:
                final_checkpoint_val_loss = val_loss
                final_checkpoint_val_bpb = val_bpb
                save_checkpoint(step, val_loss, val_bpb)

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        step_profiler = StepProfiler(args.profile_step_time, device, args.profile_inner_sections)
        ACTIVE_PROFILER = step_profiler
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_ce_loss = torch.zeros((), device=device)
        train_aux_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed and not args.use_expert_parallel:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with profile_section("data"):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with profile_section("forward"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
            train_loss += loss.detach()
            train_ce_loss += base_model.last_ce_loss if base_model.last_ce_loss is not None else loss.detach()
            train_aux_loss += base_model.last_aux_loss if base_model.last_aux_loss is not None else torch.zeros_like(loss.detach())
            with profile_section("backward"):
                (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_ce_loss /= grad_accum_steps
        train_aux_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            if isinstance(opt, Muon):
                for group in opt.param_groups:
                    group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        with profile_section("grad_sync"):
            sync_replicated_grads()
        if args.grad_clip_norm > 0:
            with profile_section("grad_clip"):
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        with profile_section("optimizer"):
            for opt_idx, opt in enumerate(optimizers):
                with profile_section(f"optimizer_{opt_idx}_{opt.__class__.__name__.lower()}"):
                    opt.step()
        zero_grad_all()
        ACTIVE_PROFILER = None
        profile_metrics = step_profiler.collect()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            route_metrics = base_model.router_metrics(reset=True)
            route_msg = ""
            if route_metrics:
                route_msg = (
                    f" route_entropy:{route_metrics['router/load_entropy_norm']:.3f}"
                    f" max_over_mean:{route_metrics['router/max_over_mean_load']:.2f}"
                    f" dead_frac:{route_metrics['router/dead_expert_frac']:.3f}"
                    f" gate_entropy:{route_metrics['router/gate_entropy']:.3f}"
                )
            profile_msg = ""
            if profile_metrics:
                profile_msg = (
                    f" prof_forward:{profile_metrics.get('time/profile/forward_ms', 0.0):.0f}ms"
                    f" prof_backward:{profile_metrics.get('time/profile/backward_ms', 0.0):.0f}ms"
                    f" prof_optim:{profile_metrics.get('time/profile/optimizer_ms', 0.0):.0f}ms"
                    f" prof_opt0:{profile_metrics.get('time/profile/optimizer_0_adam_ms', 0.0):.0f}ms"
                    f" prof_opt1:{profile_metrics.get('time/profile/optimizer_1_muon_ms', 0.0):.0f}ms"
                    f" prof_opt2:{profile_metrics.get('time/profile/optimizer_2_adam_ms', 0.0):.0f}ms"
                    f" prof_router_query:{profile_metrics.get('time/profile/router_query_ms', 0.0):.0f}ms"
                    f" prof_router_key:{profile_metrics.get('time/profile/router_key_ms', 0.0):.0f}ms"
                    f" prof_router_score:{profile_metrics.get('time/profile/router_score_ms', 0.0):.0f}ms"
                    f" prof_router_topk:{profile_metrics.get('time/profile/router_softmax_topk_ms', 0.0):.0f}ms"
                    f" prof_dispatch:{profile_metrics.get('time/profile/moe_dispatch_total_ms', 0.0):.0f}ms"
                    f" prof_pack:{profile_metrics.get('time/profile/moe_dispatch_pack_ms', 0.0):.0f}ms"
                    f" prof_gemm:{profile_metrics.get('time/profile/moe_grouped_gemm_ms', 0.0):.0f}ms"
                    f" prof_scatter:{profile_metrics.get('time/profile/moe_dispatch_scatter_ms', 0.0):.0f}ms"
                )
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ce_loss:{train_ce_loss.item():.4f} aux_loss:{train_aux_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                f"{route_msg}{profile_msg}"
            )
            if wandb_run is not None:
                wandb_payload = {
                        "train/loss": float(train_loss.item()),
                        "train/ce_loss": float(train_ce_loss.item()),
                        "train/aux_loss": float(train_aux_loss.item()),
                        "time/train_ms": approx_training_time_ms,
                        "time/step_avg_ms": approx_training_time_ms / step,
                        "optim/lr_scale": scale,
                        "optim/muon_momentum": muon_momentum,
                    }
                wandb_payload.update(route_metrics)
                wandb_payload.update(profile_metrics)
                wandb_run.log(wandb_payload, step=step)

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "memory/peak_allocated_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
                "memory/peak_reserved_mib": torch.cuda.max_memory_reserved() // 1024 // 1024,
            },
            step=step,
        )
    if not args.final_full_val or final_checkpoint_val_loss is None:
        save_checkpoint(step, final_checkpoint_val_loss, final_checkpoint_val_bpb)
    if args.skip_post_training_eval:
        if distributed:
            dist.destroy_process_group()
        if wandb_run is not None:
            wandb_run.finish()
        return

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
