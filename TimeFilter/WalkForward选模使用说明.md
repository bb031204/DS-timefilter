# SP500 选模规则滚动验证

本实验只比较选模：A 用 StockMixer 式验证损失最小，B 用验证 RankIC 最大。同一折只训练一次，模型结构和损失固定为 `SP500_2026_10_05_15_49_baseline/source_config.yaml` 的实际配置（`rank_weight=5`、`ic_weight=0`）。当前 `config.yaml` 的后来改动不会进入本实验。

在 `TimeFilter` 目录运行：

```powershell
& 'C:\Users\marti\anaconda3\envs\timefilter\python.exe' scripts/run_walkforward_selection.py --dry-run
& 'C:\Users\marti\anaconda3\envs\timefilter\python.exe' scripts/run_walkforward_selection.py
```

折、种子和逐轮 Future 诊断开关见 [walkforward_selection.yaml](walkforward_selection.yaml)。三折 Selection 均为 253 个目标日，Future 各 126 日；全部位于正式测试日 `1259–1610` 之前。每折保存 A/B 权重、选择轮次、验证期前后半段指标、逐轮 Future 指标、两套 Future 预测与指标，以及代码/数据指纹。汇总在 `outputs/walkforward_selection_analysis/<时间戳>/walkforward_summary.csv` 和 `.json`。中断后用 `--resume <该时间戳目录>` 跳过已完成的折；加 `--retry-incomplete` 可先归档未完成折再重跑。

每折的 `source_config.yaml` 是原始训练配置，`config.yaml` 是实际生效的参数快照；独立复测 `best_stockmixer_loss.pth` 或 `best_rankic.pth` 时会校验后者、时间边界和数据版本。短跑已验证独立加载 B 后复算指标与训练结束时一致。

Future 逐轮指标和相关性只用于**规则锁定后的诊断**，不参与 A/B 的权重选择。重点比较每折 `B−A` 的 RankIC、IC、MSE 和 Precision@10，而不是挑 Future 表现最高的 epoch。单种子三折只是初筛；若 B 有一致优势，再把 YAML 的 `seeds` 扩为多个种子复验。此前已查看过正式测试期，因此在该区间进一步验证新规则仍属于探索性分析。
