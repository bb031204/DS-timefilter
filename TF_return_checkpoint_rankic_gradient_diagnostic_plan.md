# TF-Return：Checkpoint 选择与梯度尺度诊断修改方案

> 目标：在不修改 TimeFilter 主干结构、GraphLearner、GCN、MoE routing 与现有训练流程主体逻辑的前提下，完成两项金融训练协议修改：
>
> 1. 将金融任务 checkpoint 的默认选择指标从 Validation MSE 改为 **Validation RankIC**，同时保留可配置选择方式；
> 2. 在训练过程中固定检查 3 次 `MSE / RankLoss / MoE Loss` 的梯度尺度，并将结果输出到日志，用于判断 `rank_weight=0.1` 和 `moe_aux_weight=0.005` 是否合理。

---

## 1. Checkpoint 改为按 Validation RankIC 选择

### 当前问题

目前已经加入：

\[
L = L_{MSE} + 0.1L_{Rank} + \lambda_{MoE}L_{MoE}
\]

但 checkpoint 仍默认按照：

\[
\min Validation\ MSE
\]

选择。

当前任务最终更关注股票横截面排序能力，因此建议将金融任务默认 checkpoint selection 改为：

\[
\boxed{\max Validation\ RankIC}
\]

### 修改要求

在 `config.yaml` 的 `training` 中明确增加/保留：

```yaml
training:
  # checkpoint 选择方式：
  # mse    -> Validation MSE 越低越好
  # IC     -> Validation IC 越高越好
  # RankIC -> Validation RankIC 越高越好
  financial_selection: RankIC
```

要求支持：

```text
mse
IC
RankIC
```

当前 SP500 金融任务默认：

```yaml
financial_selection: RankIC
```

### 实现约束

- 不修改现有金融指标计算方式；
- 不删除 MSE、IC、RankIC 等已有指标；
- checkpoint 只根据 Validation 指标选择；
- Test 不参与 checkpoint 选择；
- 推荐训练时传入 `--financial_validation_only`，确定方案后再独立测试，避免逐轮 Test 结果影响人工调参；
- 原始非金融 TimeFilter 的 EarlyStopping / checkpoint 逻辑保持不变；
- `financial_selection: mse` 时应能恢复当前 MSE 选模行为。

---

## 2. 增加 3 次损失项梯度尺度诊断

### 目的

当前训练损失：

\[
L =
L_{MSE}
+
0.1L_{Rank}
+
0.005L_{MoE}
\]

单看 loss 数值不能判断哪个目标真正主导参数更新，因此需要比较：

\[
\|\nabla L_{MSE}\|
\]

\[
\|\nabla (0.1L_{Rank})\|
\]

\[
\|\nabla (0.005L_{MoE})\|
\]

用于观察三个损失项的梯度尺度；范数比例不等于 Adam 的实际参数更新比例。

---

## 3. 诊断触发时机

每次完整训练只检查 3 次：

```text
epoch 0：正式训练开始前
epoch 1：第 1 个 epoch 完成后
epoch 5：第 5 个 epoch 完成后
```

在 `config.yaml` 中增加：

```yaml
training:
  # 0 表示训练开始前；
  # N 表示第 N 个 epoch 完成后。
  # 空列表 [] 表示关闭梯度诊断。
  gradient_diagnostic_epochs: [0, 1, 5]

  # 固定取覆盖训练期的 8 个日期；可按显存情况调至 32，与训练 batch 对照。
  gradient_diagnostic_batch_size: 8
```

---

## 4. 梯度诊断实现要求

### 4.1 固定同一个 diagnostic batch

训练启动时，从 Train Dataset 均匀取覆盖训练期的固定日期，组成一个 diagnostic batch；不从已打乱的训练 DataLoader 抽样。

之后：

```text
epoch 0
epoch 1
epoch 5
```

全部使用同一份 batch。

禁止三次检查分别随机取不同日期的数据。

### 4.2 三个梯度来自同一次 Forward

每次诊断时：

```text
固定 diagnostic batch
        ↓
执行一次 forward
        ↓
得到：
MSE
RankLoss
MoELoss
        ↓
分别计算三个加权目标的梯度范数
```

即：

\[
\nabla L_{MSE}
\]

\[
\nabla(0.1L_{Rank})
\]

\[
\nabla(0.005L_{MoE})
\]

不要为了三个 loss 分别重新 forward。

原因：当前模型包含 Dropout 和 noisy MoE routing，多次 forward 会产生不同随机状态，导致梯度不可直接比较。

### 4.3 诊断不能影响正常训练

梯度诊断必须是 observation-only。

要求：

- 不执行 `optimizer.step()`；
- 不修改模型参数；
- 不累计到正常训练 `.grad`；
- 诊断结束后清理临时梯度；
- 不改变 learning rate；
- 不改变 checkpoint selection；
- 不改变后续训练随机轨迹。

优先使用只读取梯度的方式进行计算，而不是连续对三个 loss 执行普通 `.backward()`。

### 4.4 保存并恢复随机数状态

为了保证诊断开启/关闭不会改变正常训练结果，诊断前应保存并在诊断后恢复：

```text
Python RNG state
NumPy RNG state
Torch CPU RNG state
Torch CUDA RNG state
```

诊断结束后必须恢复训练原 RNG state。

---

## 5. Gradient Norm 计算方式

统一对所有：

```text
requires_grad=True
```

的模型参数计算 global L2 gradient norm，并按 `patch_embed / backbone / moe_gate / head` 分组：

\[
\|\nabla_\theta L\|_2
=
\sqrt{
\sum_p
\|\nabla_{\theta_p}L\|_2^2
}
\]

分别得到：

```text
grad_mse
grad_rank
grad_moe
```

各组及全局都记录有梯度的参数张量数，并在该组 MSE 梯度非零时计算：

\[
ratio_{rank}
=
\frac{\|\nabla(0.1L_{Rank})\|}
{\|\nabla L_{MSE}\|}
\]

\[
ratio_{moe}
=
\frac{\|\nabla(0.005L_{MoE})\|}
{\|\nabla L_{MSE}\|}
\]

若某组 MSE 梯度为零，该组比例记为 `n/a`，不以极小分母制造夸大的数字。

---

## 6. 日志输出要求

每次梯度诊断时，在 `terminal.log` 中输出独立区块。

推荐格式：

```text
[Gradient Diagnostic]
stage=pretrain
epoch=0
batch_size=8

MSE:
  raw_loss      = ...
  weight        = 1.0
  weighted_loss = ...
  grad_norm     = ...
  grad/MSE      = 1.0

Rank:
  raw_loss      = ...
  weight        = 0.1
  weighted_loss = ...
  grad_norm     = ...
  grad/MSE      = ...

MoE:
  raw_loss      = ...
  weight        = 0.005
  weighted_loss = ...
  grad_norm     = ...
  grad/MSE      = ...
```

第 1、5 epoch 完成后分别输出：

```text
stage=post_epoch
epoch=1
```

和：

```text
stage=post_epoch
epoch=5
```

日志至少记录：

```text
raw_loss
weight
weighted_loss
grad_norm
ratio_to_mse
group
active_tensors / total_tensors
```

---

## 7. 保存结构化记录

除 `terminal.log` 外，在当前金融实验结果目录保存：

```text
gradient_diagnostics.csv
```

字段：

```text
epoch
stage
batch_size
train_indices
component
group
raw_loss
weight
weighted_loss
grad_norm
ratio_to_mse
active_tensors
total_tensors
```

该文件只用于诊断和实验记录，不参与模型训练和 checkpoint 选择。

---

## 8. 当前推荐配置

`config.yaml` 中当前金融训练相关配置建议为：

```yaml
training:
  moe_aux_weight: 0.005
  rank_weight: 0.1

  # checkpoint selection:
  # mse / IC / RankIC
  financial_selection: RankIC

  # gradient diagnostics:
  # 0 = before training
  # N = after epoch N
  # [] = disabled
  gradient_diagnostic_epochs: [0, 1, 5]
  gradient_diagnostic_batch_size: 8

  financial_seed: 2021

  batch_size: 32
  train_epochs: 100
  learning_rate: 0.0001
```

训练目标保持：

\[
\boxed{
L=
L_{MSE}
+
0.1L_{Rank}
+
0.005L_{MoE}
}
\]

checkpoint 默认按照：

\[
\boxed{
\max Validation\ RankIC
}
\]

选择。

---

## 9. 保持不变的部分

本次修改禁止改变：

- TimeFilter backbone；
- PatchEmbed；
- GraphLearner；
- GCN；
- MoE hard routing；
- 图节点定义；
- StockMixer-style RankLoss 数学形式；
- SP500 数据划分；
- IC / RankIC / Precision@10 / Sharpe 等现有评价实现。

本次只新增：

1. `financial_selection` 的配置化与默认 `RankIC`；
2. 固定 3 次的 gradient-scale diagnostic；
3. 诊断日志与 CSV 保存；参数组范数和有梯度参数数一起记录。

目标是改善金融实验协议与可解释性，不改变 TimeFilter 模型结构。
