# Params-Aware MoE

参数感知路由 MoE（PA-MoE）实验代码与结果，基于 `parameter-golf`。PA router 从专家参数构造 key，并将 hidden state 投影为 query；Linear router 作为匹配对照。

当前快照包含 **2026-05-31 训练源码**、**2026-06-05 full-validation 结果**和 **2026-11-19 更新的实验报告**。主实验为 FineWeb-Edu 10B tokens、单种子、16 个 PA/Linear 配对配置；结果是阶段性结论，不是多种子统计结论。

## 当前结果

- 在 E=32/64/128/256、LB=0/0.01 的 8 组配对比较中，PA 的 BPB 均低于 Linear，优势为 **0.0186–0.0263 BPB**。
- 最佳配置为 PA E256、top-16、product dim=256、LB=0.01：**1.684696 CE / 0.994241 BPB**。
- 完整结论、实验限制和下一步见 [实验报告](docs/experiment_report.md)（[PDF](docs/experiment_report.pdf)）。
- 可分析的指标、路由张量、图表和最终训练日志见 [结果索引](results/README.md)。

## 目录结构

```text
.
├── train_gpt.py                 # CUDA/DDP 训练入口；当前为完整历史快照
├── train_gpt_mlx.py             # MLX 入口
├── analysis/
│   └── analyze_routes.py        # 从 checkpoint 采集路由统计
├── scripts/train/               # 10B 主实验的原始训练配置与串行作业
├── data/                        # 数据下载/导出工具；数据和 tokenizer 不入库
├── docs/                        # 实验报告
└── results/
    ├── finewebedu_10b_20260605/ # 最终 full-val 结果（主结果）
    └── route_analysis_2m_20260531/ # 早期 2M-token 路由分析
```

`train_gpt.py` 暂时保持单文件实现，以确保历史实验配置可直接对应源码。后续重构建议拆分模型、router、优化器和评估模块，但保留此入口及 `scripts/train/` 中的环境变量接口；已发布结果目录只追加新实验，不覆盖旧结果。

## 环境与数据

```bash
uv venv
uv pip install -r requirements.txt
```

主实验要求以下本地资源（大二进制文件不在 GitHub 仓库中）：

```text
data/datasets/finewebedu10B_sp1024/fineweb_train_*.bin
data/datasets/finewebedu10B_sp1024/fineweb_val_*.bin
data/tokenizers/fineweb_1024_bpe.model
```

数据集来源、token 数和 shard 信息记录在 [`dataset_manifest.json`](results/finewebedu_10b_20260605/dataset_manifest.json)。通用数据工具见 [`data/README.md`](data/README.md)；使用前应确认下载的数据集与主实验的 FineWeb-Edu `sample-10BT`、sp1024 导出一致。

## 训练与分析

默认 Dense 基线：

```bash
python train_gpt.py
```

主实验使用双卡 DDP。训练脚本会先切换到仓库根目录，因此可从任意工作目录调用：

```bash
bash scripts/train/run_pa_lb0_pd256_e64_e128_e256.sh
bash scripts/train/run_after_dense_m4_linear128_256_pa32pd128.sh
```

这些脚本会串行启动多个完整 10B-token 作业，不是快速 smoke test。可用 `PYTHON_BIN` 覆盖默认的 `.venv/bin/python`；具体环境变量直接保存在各脚本中。

对已有 checkpoint 做 2M-token 路由分析：

```bash
python -m analysis.analyze_routes \
  --checkpoint-dir /path/to/checkpoints \
  --data-path data/datasets/finewebedu10B_sp1024 \
  --out-dir /path/to/output \
  --device cuda:0
```

注意：当前快照包含上述 2M-token 分析脚本，但没有单独保存 2026-06-05 full-val 汇总/绘图脚本；最终指标、逐层指标和路由张量均已保留，不能据此声称一键复现全部图表。

## 协作约定

- 修改训练逻辑放在根目录训练入口或后续拆分出的源码模块中；历史作业配置放在 `scripts/train/`。
- 新实验使用新的 `results/<数据_预算_日期>/` 目录，并附配置、指标来源和 checkpoint 外部存储位置。
- 不提交 checkpoint、数据集、tokenizer、临时压缩包、PID 文件或 W&B 本地文件。原始大产物仍保留在本地 `runpod_results`，本仓库保留其轻量、可协作的部分。
