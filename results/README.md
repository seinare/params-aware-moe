# 实验结果索引

## 主结果：`finewebedu_10b_20260605/`

2026-06-05 对 16 个模型统一执行 full validation：99,617,792 tokens，评估时 auxiliary loss 系数为 0。主实验为 `2 router types × 4 expert counts × 2 LB coefficients`，所有配置均训练 19,073 updates、约 10B tokens，seed=1337。

- `full_val_summary.csv` / `.jsonl`：最终 CE、BPB、评估吞吐及模型配置；报告数值以此为准。
- `layer_route_metrics.csv`：16 个模型的逐层路由指标。
- `route_tensors/`：逐专家 gate、probability、selection 等轻量张量，可用于重新绘图。
- `resolved_checkpoints.csv`：模型与原始 checkpoint 的映射。
- `training_runs.json`：从训练日志提取的训练曲线和摘要，包含 2 个历史 Dense 参考；其中的验证摘要可能包含 auxiliary loss，不是最终 full-val 的替代来源。
- `logs/`：主实验 16 个最终训练日志和 2 个 Dense 参考日志。
- `figures/`：9 张图的 PNG/SVG/PDF、参数对照表和静态浏览页。
- `dataset_manifest.json`：FineWeb-Edu 数据来源与导出信息。

CSV/JSONL 中的 `checkpoint` 和 `route_tensor` 字段保留 RunPod 原始路径，用于追溯，不是仓库内可直接访问的路径。本目录中的路由张量统一位于 `route_tensors/`。模型 checkpoint 未上传。

最佳结果：`pa_e256k16_pdim256_lb0.01`，CE=1.684696，BPB=0.994241。详细分析见 [`../docs/experiment_report.md`](../docs/experiment_report.md)。

## 早期分析：`route_analysis_2m_20260531/`

2026-05-31 使用 2,097,152 个训练 token 采集的初步路由统计，覆盖当时可用的部分 checkpoint。它用于追溯分析过程，不是报告中的最终 full-validation 结果。

- `results_2m/summary.csv` / `.json`：模型级汇总。
- `results_2m/*_layer_metrics.csv`：逐层指标。
- `results_2m/*.pt`：对应路由张量。
- `dashboard/`：静态结果浏览页及其数据。
- `route_analysis_2m_20260531_150937.log`：原始分析日志。

## 未纳入 GitHub 的本地产物

原始目录约 7.6 GB。以下内容仍保留在本地 `runpod_results`，未删除：

- `checkpoint/`：约 5.5 GB 模型和优化器状态。
- `data/`、`incoming/`：数据集、tokenizer、重复解包和传输压缩包。
- `artifacts/`：约 82 MB 其他二进制产物。
- `experiment_figures_png/`：与主结果 `figures/` 中 PNG 重复的导出。
- `logs/` 中 UUID 调试日志、性能试验、失败重启、PID/HTTP 服务文件等非主结果记录。
- `runpod_code_20260531.tar`、`runpod_logs_20260531.tar.gz`：已被源码目录或筛选后日志替代的归档。
