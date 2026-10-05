# Finance-TimeFilter：P0 / P1 金融适配修改方案

> 目标：在**不破坏原始 TimeFilter 主干**的前提下，对金融任务做系统化适配。  
> 所有新增能力必须采用**模块化、可开关、可回退**设计，并在 `config.yaml` 中增加独立 `finance_adaptation` 配置段，便于后续消融实验和回退。

---

## 1. 总体原则

TimeFilter 核心主干继续保留：

```text
Patch Embedding
→ GraphLearner
→ MoE Relation Filtering
→ GCN
→ FFN / GraphBlock
→ Prediction Head
```

金融适配只在输入、位置编码、图 mask、输出头和损失层增加可选模块：

```text
EOD5
→ [Finance Input Adapter]
→ [Financial Patch Projection]
→ TimeFilter Graph Backbone
→ [Finance Output Head]
→ Financial Loss
```

要求：

- 不删除原 `returns` 输入路径；
- 不删除原 EOD5 naive 路径；
- 所有新增模块必须能独立开关；
- `finance_adaptation.enabled=false` 时尽量恢复当前金融 baseline；
- 非金融数据集不受影响；
- 代码一次搭好完整模块，但实验仍按模块逐步消融。

---

# 2. 建议新增独立配置段

```yaml
finance_adaptation:
  enabled: true

  input_adapter:
    enabled: true
    type: stockmixer_lite

    input_layernorm: true

    indicator_mixer:
      enabled: true
      hidden_dim: 16
      residual: true

    temporal_mixer:
      enabled: true
      hidden_dim: 16
      residual: true

  positional_encoding:
    mode: patch_only
    # original / patch_only / none

  graph:
    strict_masked_softmax: true

  output_head:
    type: direct_return
    # direct_return / price_anchor

  loss:
    rank_weight: 5.0

    ic_loss:
      enabled: false
      weight: 0.0

```

---

# 3. P0：优先完成

## P0-1：Financial Input Adapter

### 当前问题

当前 EOD5：

```text
[B,16,474,5]
→ 每股每 patch 的 8×5 展平
→ Linear(40,d_model)
```

虽然维度正确，但没有显式建模：

```text
Indicator relation
Temporal relation
```

StockMixer 值得借鉴的是：

```text
LayerNorm
Indicator Mixing
Temporal Mixing
```

而不是它的 Stock Mixing，因为 TimeFilter 已经有 GraphLearner + GCN 负责跨股票建模。

### 建议结构

```text
[B,16,474,5]
→ [B,474,16,5]
→ Input LayerNorm
→ Indicator Mixer
→ Temporal Mixer
→ [B,474,2,8,5]
→ flatten 8×5
→ Linear(40,d_model)
→ [B,948,d_model]
→ TimeFilter Backbone
```

### Indicator Mixer

只沿 5 个特征维进行轻量 MLP：

```text
5 → 16 → 5
```

采用 residual：

```text
x = x + IndicatorMLP(LN(x))
```

### Temporal Mixer

在单只股票内部沿时间维做轻量 mixing，可先使用：

```text
8 → 16 → 8
```

或：

```text
16 → 16 → 16
```

第一版优先 patch 内 `8→16→8`，减少参数量。

### 必须支持独立开关

```yaml
input_layernorm: true/false

indicator_mixer:
  enabled: true/false

temporal_mixer:
  enabled: true/false
```

这样可做：

```text
Raw EOD5
→ +LayerNorm
→ +Indicator Mixer
→ +Temporal Mixer
```

逐步消融。

---

## P0-2：重新校准 RankLoss

### 当前问题

最新 EOD5：

```text
rank_weight = 0.1
```

Epoch 5 梯度诊断：

```text
||∇(0.1 × RankLoss)|| / ||∇MSE|| ≈ 0.006
```

约只有 MSE 梯度的 0.6%，排序目标几乎没有参与训练。

### 修改思路

保留：

```text
Loss =
MSE
+ rank_weight × RankLoss
+ moe_aux_weight × MoE Loss
```

但将 `rank_weight` 作为正式金融适配参数。

建议先做梯度探针：

```text
1 / 5 / 10
```

第一候选建议：

```text
rank_weight = 5
```

因为在当前固定状态下大约对应：

```text
Rank gradient ≈ 0.3 × MSE gradient
```

注意这是静态估计，完整训练后比例会变化。

### 与选模规则解耦

训练权重：

```yaml
finance_adaptation:
  loss:
    rank_weight: 5.0
```

StockMixer-style 选模仍可固定：

```yaml
training:
  financial_selection: stockmixer_val_loss
  stockmixer_selection_rank_weight: 0.1
```

即：

```text
训练目标可加强 RankLoss
但 checkpoint selection 仍保持固定协议
```

也可以另开一条 `Val RankIC` 选模实验，但不要混在同一对照中。

---

## P0-3：输入端 Normalization

### 当前问题

EOD5 当前：

```text
raw EOD5
→ Linear
```

且：

```yaml
norm: false
```

这避免了旧 RevIN 的均值恢复 shortcut，但也意味着五特征没有输入端 normalization。

### 修改思路

只做输入 normalization：

```text
EOD5
→ Input LayerNorm
→ Finance Adapter
→ TimeFilter
→ Return
```

不要恢复旧流程：

```text
Normalize
→ Predict
→ Denormalize
→ + historical mean
```

输出 return 不做历史均值恢复。

### 第一版建议

```yaml
input_layernorm: true
```

后续可再预留：

```text
train-stat scale-only normalization
```

但第一版不要同时实现太多方案。

---

## P0-4：Strict Masked Softmax

### 当前问题

当前：

```text
adj = adj * mask
→ 被删边 = 0
→ softmax(adj)
```

由于：

```text
exp(0)=1
```

被删除边会重新获得非零概率。

### 修改思路

改为：

```text
adj logits
→ invalid edge = -inf
→ softmax
```

确保：

```text
masked edge probability = 0
```

必须保留 self-loop，避免某行全部为 `-inf`。

### 配置

```yaml
finance_adaptation:
  graph:
    strict_masked_softmax: true
```

关闭时恢复原实现。

---

## P0-5：Patch-only Positional Encoding

### 当前问题

当前 PE 实际编码：

```text
Stock0-P0
Stock0-P1
Stock1-P0
Stock1-P1
...
```

相当于同时编码股票编号和 patch 位置。

股票编号通常没有自然序关系。

### 修改思路

只编码 patch：

```text
Patch0 → PE0
Patch1 → PE1
```

所有股票共享：

```text
Stock_i-P0 → PE0
Stock_i-P1 → PE1
```

### 配置

```yaml
finance_adaptation:
  positional_encoding:
    mode: patch_only
```

支持：

```text
original
patch_only
none
```

---

# 4. P1：第二阶段修改

## P1-1：轻量化 Backbone

### 当前问题

当前：

```text
d_model=512
d_ff=2048
e_layers=2
```

约 4.8M 参数，而 SP500 只有约 990 个训练目标日。

### 建议

金融适配版默认候选：

```yaml
d_model: 256
d_ff: 1024
e_layers: 1
n_heads: 4
```

仍完整保留：

```text
GraphLearner
MoE
GCN
FFN
```

因此不破坏 TimeFilter 主体。

### 配置

```yaml
finance_adaptation:
```

关闭时使用原 `model` 配置。

---

## P1-2：Price-Anchored Output Head

### 目的

借鉴 StockMixer 的：

```text
prediction
→ base price
→ return
```

思路，但不直接复制完整 StockMixer。

### 两种可切换输出头

#### direct_return

```text
latent → Linear → next-day return
```

#### price_anchor

```text
latent
→ z
→ P_hat(t+1) = P_t × (1 + z)
→ return = (P_hat(t+1)-P_t)/P_t
```

训练仍在 return 空间计算：

```text
MSE
RankLoss
```

### 配置

```yaml
finance_adaptation:
  output_head:
    type: direct_return
```

可切：

```text
direct_return
price_anchor
```

该模块主要用于消融，不应强制开启。

---

## P1-3：可选 IC Loss

### 目的

让训练目标更直接服务于横截面 IC。

预留：

```yaml
finance_adaptation:
  loss:
    ic_loss:
      enabled: false
      weight: 0.0
```

后续实现：

```text
IC Loss = 1 - Pearson(pred, target)
```

按每个交易日、474 只股票的横截面计算。

最终可扩展：

```text
Loss =
MSE
+ λ_rank × RankLoss
+ λ_ic × IC Loss
+ λ_moe × MoE Loss
```

第一版不要默认开启 IC Loss。

---

# 5. 模块化目录建议

建议新增独立金融模块目录，例如：

```text
layers/
└── finance/
    ├── input_adapter.py
    ├── indicator_mixer.py
    ├── temporal_mixer.py
    ├── financial_patch_embed.py
    ├── financial_positional_encoding.py
    ├── financial_head.py
    └── masked_softmax.py
```

原 `models/TimeFilter.py` 只负责：

```text
读取 config
→ 判断是否启用金融适配
→ 调用对应模块
```

不要把所有金融逻辑直接堆入 `TimeFilter.py`。

---

# 6. Forward 路径建议

```text
if finance_adaptation.enabled == false:
    → 原 TimeFilter 路径

else:
    if input_features == returns:
        → 保留 returns 路径

    if input_features == eod5:
        → optional Input LayerNorm
        → optional Indicator Mixer
        → optional Temporal Mixer
        → Financial Patch Projection

    → optional patch-only PE

    → 原 TimeFilter Backbone

    → optional direct_return / price_anchor head

    → financial loss
```

注意：

```text
eod5
```

不能自动等同于：

```text
所有金融模块全部开启
```

必须允许：

```text
EOD5 raw
EOD5 + LN
EOD5 + Indicator
EOD5 + Indicator + Temporal
...
```

---

# 7. 推荐 Finance-TimeFilter v1 配置

```yaml
forecast:
  seq_len: 16
  pred_len: 1
  input_features: eod5

model:
  d_model: 256
  d_ff: 1024
  e_layers: 1
  n_heads: 4
  patch_len: 8
  alpha: 0.2
  top_p: 0.5
  dropout: 0.1

finance_adaptation:
  enabled: true

  input_adapter:
    enabled: true
    type: stockmixer_lite
    input_layernorm: true

    indicator_mixer:
      enabled: true
      hidden_dim: 16
      residual: true

    temporal_mixer:
      enabled: true
      hidden_dim: 16
      residual: true

  positional_encoding:
    mode: patch_only

  graph:
    strict_masked_softmax: true

  output_head:
    type: direct_return

  loss:
    rank_weight: 5.0
    ic_loss:
      enabled: false
      weight: 0.0

```

第一版暂时不要同时改：

```text
MoE hard routing
Signed Graph
行业图
外部关系图
```

---

# 8. 消融实验必须可支持

| 实验 | LN | Indicator | Temporal | Patch-only PE | Strict Mask | Lightweight | Rank Weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive EOD5 | OFF | OFF | OFF | OFF | OFF | OFF | 0.1 |
| + LN | ON | OFF | OFF | OFF | OFF | OFF | 0.1 |
| + Indicator | ON | ON | OFF | OFF | OFF | OFF | 0.1 |
| + Temporal | ON | ON | ON | OFF | OFF | OFF | 0.1 |
| + Rank | ON | ON | ON | OFF | OFF | OFF | 5 |
| + PE Fix | ON | ON | ON | ON | OFF | OFF | 5 |
| + Graph Fix | ON | ON | ON | ON | ON | OFF | 5 |
| + Lightweight | ON | ON | ON | ON | ON | ON | 5 |

代码需要支持这些组合，但不要求一次全部跑完。

---

# 9. 回退要求

以下配置：

```yaml
finance_adaptation:
  enabled: false
```

应恢复当前原有金融 TimeFilter 行为。

各子模块也必须能独立关闭：

```yaml
finance_adaptation:
  input_adapter:
    enabled: false

  positional_encoding:
    mode: original

  graph:
    strict_masked_softmax: false

  output_head:
    type: direct_return
```

任何单个模块无效时，都应通过配置回退，而不是删除代码。

---

# 11. 最终目标结构

```text
EOD5
  ↓
Input LayerNorm
  ↓
Indicator Mixer
  ↓
Temporal Mixer
  ↓
Patch Projection
  ↓
Patch-only PE
  ↓
========================
原 TimeFilter 核心主干
GraphLearner
MoE
GCN
FFN
========================
  ↓
Financial Head
  ↓
Next-day Return
  ↓
MSE + RankLoss (+ optional IC Loss)
```

核心定位：

> 借鉴 StockMixer 在金融输入建模、归一化和轻量化方面的优势，但保留 TimeFilter 的图关系学习主干，使整体成为“Financial Adapter + TimeFilter Graph Backbone”，而不是把 StockMixer 与 TimeFilter 简单拼接。
