# TF-Return 第一阶段修改方案（SP500）

> 目标：在**不修改 TimeFilter 原有模型结构和核心代码逻辑**的前提下，修正当前 `TF-return` 在 SP500 下一日收益率预测中的主要配置与训练错配，得到一个稳定、可复现的金融基线。
>
> 当前阶段只处理：**Patch、Normalize、训练目标、模型容量、图稀疏度、MoE 辅助损失权重**。
>
> 不修改：`PatchEmbed` 结构、`GraphLearner` 数学逻辑、`GCN`、MoE hard routing、节点定义、TimeFilter backbone。

---

## 1. 修改原则

1. **保留原始 TimeFilter 行为**：原始配置必须仍可运行和复现；金融适配通过独立配置项或 finance-only 逻辑启用。
2. **只使用当前 Return 输入**：输入仍为过去 16 个交易日的股票收益率，本阶段不引入 OHLCV 或新的特征编码模块。
3. **优先使用现有超参数调节**：`patch_len`、`d_model`、`d_ff`、`e_layers`、`alpha`、`top_p`、`dropout`、`moe_aux_weight`。
4. **参数选择只看 Validation**：Test 不参与超参数选择和 checkpoint 选择。

---

# 2. 当前第一阶段需要修改的问题

## 2.1 Patch=16 导致 Temporal / Spatial-Temporal 图退化

### 问题

当前：

```yaml
seq_len: 16
patch_len: 16
```

每只股票只有 1 个 patch，因此 Spatial 存在，但 Temporal 和 Spatial-Temporal 关系为空，TimeFilter 的多 patch 时空图机制发生退化。

### 修改

不修改 PatchEmbed 或图结构，只调整已有 `patch_len`：

```text
patch_len = 16   # 原始 baseline
patch_len = 8    # 2 patches / stock，优先候选
patch_len = 4    # 4 patches / stock，补充候选
```

当前优先使用：

```yaml
patch_len: 8
```

理由：恢复 Temporal / Spatial-Temporal 关系，同时图规模仍相对可控。

---

## 2.2 Normalize 对股票收益率可能不适配（已经修改）

### 问题

TimeFilter 当前执行窗口内减均值、除标准差，并在预测后重新乘标准差、加回历史均值。对于近零均值的股票日收益率，这可能引入历史均值偏置。

### 修改

保留原 `Normalize` 实现，仅增加 finance-only 控制开关：

```text
Norm ON  = 完全保持原 TimeFilter 行为
Norm OFF = 金融任务中绕过 normalize / denormalize
```

要求：

- 原始 TimeFilter 默认仍为 `Norm ON`；
- finance-only 配置可以明确关闭；
- 不重写 Normalize 模块内部公式。

当前优先候选：

```yaml
model:
  norm: false
```

已接入金融配置：`false` 使用现有 `Normalize(non_norm=True)`，同时绕过归一化和反归一化；`true` 保持原始行为。非金融任务及未显式配置该项的旧命令仍默认开启归一化。此处是待验证候选，不能仅凭收益率均值接近零就断定关闭更优。

---

## 2.3 训练目标与金融评价目标不匹配（已经修改）

### 问题

修改前 TimeFilter 主要优化：

\[
L=MSE+\lambda_{MoE}L_{MoE}
\]

但金融任务更关注 IC、RankIC、Precision@10、Sharpe。单独 MSE 不直接约束同一天股票之间的横截面排序。

### 修改：沿用 StockMixer 的 RankLoss

采用 StockMixer 的训练目标思想：

\[
L_{financial}=L_{MSE}+\alpha_{rank}L_{rank}
\]

第一版固定：

\[
\alpha_{rank}=0.1
\]

完整 TimeFilter 金融损失：

\[
L=L_{MSE}+0.1L_{rank}+\lambda_{MoE}L_{MoE}
\]

其中：

- `MSE`：控制收益率数值误差；
- `RankLoss`：约束同一天股票之间的相对收益排序；
- `MoE Loss`：保留 TimeFilter 原有辅助训练目标。

### RankLoss 使用要求

TimeFilter 已直接输出下一日收益率，因此：

- 不需要 StockMixer 的“预测价格 → 收益率”转换；
- 直接对预测收益率和真实收益率计算 RankLoss；
- 必须逐交易日、沿股票维度计算；
- 禁止将多个交易日 flatten 后一起计算排序。

当前固定：

```yaml
training:
  rank_weight: 0.1
  moe_aux_weight: 0.005
```

已接入金融训练：沿每个目标日的股票维度分别计算 StockMixer 形式的成对 hinge loss；SP500 全部股票有效，其他金融数据集沿用现有股票掩码。`rank_weight: 0` 恢复原 MSE + MoE 目标，非金融任务仍使用原损失。训练日志另记原始及加权后的 RankLoss；checkpoint 仍按 `training.financial_selection` 指定的验证集指标选择，当前值为 `mse`。`0.1` 仅为待验证候选，需结合损失及梯度尺度分析。

---

## 2.4 模型容量过大

### 问题

当前：

```yaml
d_model: 512
d_ff: 2048
e_layers: 2
```

模型约 4.8M 参数，而 SP500 训练目标日约 990 个，且相邻 16 日窗口高度重叠。已有实验 best epoch 集中在 4～5 epoch，说明当前模型存在明显的快速拟合和泛化压力。

### 修改

不改变网络层类型，仅使用原有超参数缩小模型。

优先候选：

```text
128 / 512 / 1 layer
256 / 1024 / 1 layer
512 / 2048 / 2 layers  # 原始对照
```

保持：

\[
d_{ff}\approx4d_{model}
\]

当前主推荐：

```yaml
d_model: 128
d_ff: 512
e_layers: 1
```

---

## 2.5 MoE hard routing 不可由预测损失直接训练

### 问题

当前 hard routing 存在不可微问题。

### 本阶段处理

**不修改 MoE routing 结构。**

保持原始 TimeFilter MoE，只控制已有辅助损失权重：

```yaml
moe_aux_weight: 0.005
```

保留以下值用于对照：

```text
0
0.005
0.05
```

本阶段不引入新的 routing 机制。

---

## 2.6 图过密（已经修改）

### 问题

修改前：

```yaml
alpha: 0.1
```

会保留大部分候选边。对于 SP500 474 只股票，过密图可能引入无关股票噪声、over-smoothing，并稀释局部有效关系。

### 修改

不改变 GraphLearner、mask、GCN 结构，只调整原有 `alpha`。

候选：

```text
alpha = 0.5
alpha = 0.7
alpha = 0.8
```

当前主推荐：

```yaml
alpha: 0.7
```

已将当前金融 `config.yaml` 的 `model.alpha` 设为 `0.7`；仅改变原有图筛选参数，未修改 GraphLearner、mask、GCN 或 MoE 路由。它是待验证的候选值，稀疏化效果和指标改善尚未通过完整训练确认。

`top_p` 暂时保持：

```yaml
top_p: 0.5
```

不要同时调整 `alpha` 和 `top_p`。

---

# 3. SP500 推荐候选参数组合

> 以下参数是当前第一阶段优先验证的候选组合，不预先视为最终最优配置。

所有方案统一：

```yaml
seq_len: 16
pred_len: 1
n_heads: 4
top_p: 0.5
dropout: 0.1
rank_weight: 0.1
moe_aux_weight: 0.005
learning_rate: 0.0001
```

## 方案 A：主推荐方案

```yaml
seq_len: 16
patch_len: 8

d_model: 128
d_ff: 512
e_layers: 1
n_heads: 4

alpha: 0.7
top_p: 0.5

dropout: 0.1
norm: OFF

rank_weight: 0.1
moe_aux_weight: 0.005

learning_rate: 0.0001
batch_size: 4~8
train_epochs: 100
```

特点：

- 2 patches / stock；
- 恢复 Temporal / Spatial-Temporal 图；
- 显著减小模型容量；
- 图明显更稀疏；
- 关闭历史均值恢复；
- 使用 StockMixer-style RankLoss。

---

## 方案 B：中等容量方案

```yaml
seq_len: 16
patch_len: 8

d_model: 256
d_ff: 1024
e_layers: 1
n_heads: 4

alpha: 0.5
top_p: 0.5

dropout: 0.1
norm: OFF

rank_weight: 0.1
moe_aux_weight: 0.005

learning_rate: 0.0001
batch_size: 4
train_epochs: 100
```

特点：

- 保留更多模型容量；
- 图比方案 A 更稠密；
- 用于判断 `d_model=128` 是否压缩过度。

---

## 方案 C：更细时间粒度方案

```yaml
seq_len: 16
patch_len: 4

d_model: 64~128
d_ff: 256~512
e_layers: 1
n_heads: 4

alpha: 0.8
top_p: 0.5

dropout: 0.1
norm: OFF

rank_weight: 0.1
moe_aux_weight: 0.005

learning_rate: 0.0001
batch_size: 1~2
train_epochs: 100
```

特点：

- 4 patches / stock；
- 时间关系更丰富；
- 图节点数明显增加；
- 通过更小模型、更高 `alpha` 和更小 batch 控制显存。

---

# 4. 原始 Baseline 必须保留

原始配置单独保留，不能被金融配置覆盖：

```yaml
seq_len: 16
patch_len: 16

d_model: 512
d_ff: 2048
e_layers: 2
n_heads: 4

alpha: 0.1
top_p: 0.5
dropout: 0.1

norm: ON

loss:
  mse: true
  rank_weight: 0
  moe_aux_weight: 0.05
```

定义为：

```text
TF-return Original Baseline
```

用于保证原始 TimeFilter 路径和当前金融修改均可复现。

---

# 5. 实现约束

本阶段允许：

- 调整现有超参数；
- 增加 finance-only `norm` 开关；
- 增加 finance-only StockMixer-style RankLoss；
- 增加 `rank_weight`；
- 调整金融 checkpoint selection 配置；
- 保存完整实验配置和指标。

本阶段禁止：

- 修改 `PatchEmbed` 结构；
- 修改 `GraphLearner` 数学逻辑；
- 替换 GCN；
- 修改 MoE hard routing；
- 修改 TimeFilter backbone；
- 改变图节点定义；
- 新增 attention；
- 新增 Signed Graph；
- 引入新的多特征编码网络。

---

# 6. 第一阶段目标配置

以下是后续候选目标，**并非本轮全部已实现的配置**；模型容量与 batch size 尚未调整：

```yaml
forecast:
  seq_len: 16
  pred_len: 1

model:
  patch_len: 8
  d_model: 128
  d_ff: 512
  e_layers: 1
  n_heads: 4
  alpha: 0.7
  top_p: 0.5
  dropout: 0.1
  norm: false

training:
  rank_weight: 0.1
  moe_aux_weight: 0.005
  learning_rate: 0.0001
  batch_size: 4~8
  train_epochs: 100
```

训练损失：

\[
\boxed{L=L_{MSE}+0.1L_{rank}+0.005L_{MoE}}
\]

评价统一保留：

```text
MSE
MAE
IC
RIC
RankIC
RankICIR
Precision@10
Sharpe5
Directional Accuracy
```

参数与 checkpoint 选择只使用 Validation，Test 不参与任何调参。

## 6.1 本轮实际配置（已经修改）

本轮仅启用 `model.norm: false`、`training.rank_weight: 0.1`、`model.alpha: 0.7`。`patch_len: 8`、`d_model: 512`、`d_ff: 2048`、`e_layers: 2`、`batch_size: 32`、`moe_aux_weight: 0.005` 均沿用修改前的当前配置；尚未开展新的完整训练，不能把本轮代码验证视为预测效果验证。常规金融实验从 `TimeFilter/config.yaml` 读取，命令行同名参数可覆盖配置。开发对照时可传 `--financial_validation_only` 避免每轮查看 Test，再用选定的最佳权重进行独立测试。
