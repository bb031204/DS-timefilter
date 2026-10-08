# SP500 历史选模审计

此审计沿用 TimeFilter 主模型，在每个历史 Fold、每个随机种子只训练一次；四条规则从**同一训练轨迹**用验证集选权重，再到紧随其后的历史 Future 区间比较。它是研究选模可靠性的独立入口，不改变普通 `scripts/run_financial.py` 的行为，也不读取当前 `config.yaml`。

冻结配置见 `walkforward_model_selection.yaml`。默认引用 2026-10-07 的 `ic_weight=0.01`、`batch_size=8`、`rank_weight=10` 实验快照。A 选验证集 StockMixer 损失最低，B 选验证 IC 最高，C 选验证 RankIC 最高，D 选验证集前后两段 RankIC 均值减去 `0.5 ×` 两段总体标准差最高。前三个 Fold 的 Future 都在正式测试起点 1259 之前。Future 不参与**同一 Fold 的权重选择**；四规则比较会使用这些 Future 结果，因此它们属于历史元验证。

事先固定两个主观察量：各折 Future 的 RankIC 与 Top5 相对全市场的等权日均超额收益；IC、Precision@10 和 SR(top5) 作为代价检查。三折只做描述性初筛，不能自动宣布赢家。

在 `TimeFilter` 目录使用 `timefilter` 解释器：

```powershell
& 'C:\Users\marti\anaconda3\envs\timefilter\python.exe' .\scripts\run_walkforward_model_selection.py --dry-run
& 'C:\Users\marti\anaconda3\envs\timefilter\python.exe' .\scripts\run_walkforward_model_selection.py
```

完整运行会顺序训练 3 Fold × 1 Seed。输出在 `outputs/walkforward_model_selection_analysis/<时间戳>/`；根目录有按 Fold、Seed、Rule 汇总的 CSV/JSON，各 Fold 内保存四个 `best_rule_*.pth`、`selection_summary.json`、`future_metrics_rule_*.json`、预测与逐日 Top5 收益。默认 `future_each_epoch: false`，只评估选中的四份权重；改为 `true` 可额外记录逐轮 Future 诊断，但仍不能用它挑 checkpoint。

中断后按实际输出目录恢复：

```powershell
& 'C:\Users\marti\anaconda3\envs\timefilter\python.exe' .\scripts\run_walkforward_model_selection.py --resume .\outputs\walkforward_model_selection_analysis\20261008_120000 --retry-incomplete
```

将示例时间戳换成实际生成的目录名。

恢复会核对 plan、冻结配置、代码和数据版本。结果只做描述性比较；三个历史 Fold 和一个种子不足以自动宣布最优规则。此前正式测试期已被反复查看，不能再称其为完全未触及的最终测试。若要对其他训练配置做审计，请另建 plan 指向其冻结 `source_config.yaml`，避免混合实验。回退到原行为只需继续使用 `scripts/run_financial.py`；新增功能仅在新 runner 的专用开关下启用。
