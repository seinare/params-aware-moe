# PA-MoE experiment figures

来自 16 模型 full-validation 对照和对应训练日志。每张图提供 300-DPI PNG、SVG 和 PDF；`index.html` 可静态浏览，`param_comparison.csv` 与 `dense_param_references.csv` 保存图中参数对照数据。

- `fig1_final_bpb_vs_experts`：不同专家数的最终 BPB，含历史 Dense 参考。
- `fig2_pa_gain_vs_experts`：相同专家数下 PA 相对 Linear 的 BPB 增益。
- `fig3_lb_routing_effect`：load balancing 对选择熵和集中度的影响。
- `fig4_bpb_vs_route_concentration`：最终 BPB 与专家负载集中度。
- `appendix_layerwise_routing_heatmap`：各模型、各层的路由集中度。
- `appendix_sorted_gate_distribution`：E256 的排序平均 gate mass。
- `appendix_sorted_select_distribution`：层内排序后跨层平均的 selected-expert 分布。
- `appendix_eval_throughput`：采集路由统计时的 full-val 吞吐，不代表纯推理吞吐。
- `appendix_training_curves`：从训练日志提取的 CE 曲线。

数据源为本目录上级的 `full_val_summary.csv`、`layer_route_metrics.csv`、`route_tensors/` 和 `training_runs.json`。Full validation 为 99,617,792 tokens；最终 CE/BPB 不含 auxiliary loss。当前源码快照未包含这些图的生成脚本。
