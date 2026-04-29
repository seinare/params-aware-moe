# Params-Aware MoE

Bootstrap for experiments derived from `parameter-golf`.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

Training defaults expect the Parameter Golf FineWeb export:

```bash
python train_gpt.py
```

On `xjhl4`, this project is intended to live under `/data1/xjh/code/params-aware-moe` with `data/datasets` and `data/tokenizers` symlinked to the sibling `/data1/xjh/code/parameter-golf/data` assets.
