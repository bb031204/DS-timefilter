# TimeFilter 金融任务：当前修改优先级

> 目标：先解决影响实验公平性、训练目标有效性和基础泛化能力的问题，再进行图结构与 MoE 机制改造。  
> 当前阶段不应继续优先微调 `alpha` 或直接加入 Signed Graph。

---

## P0：最高优先级

### 1. 输入协议对齐：Return1 vs EOD5

当前 TimeFilter 只输入单维收益率：

```text
[B, 16, 474]
```

而 StockMixer 使用 `SP500.npy` 的 5 维 EOD 输入，因此两者目前不是同信息量比较。

建议增加金融专用输入切换：

```text
input_mode: return1 / eod5
```

其中：

```text
return1:
[B, 16, 474]

eod5:
[B, 16, 474, 5]
```

EOD5 不要直接将 5 个特征展开成 `474×5=2370` 个变量。

应保持：

```text
每只股票 = 一个图实体
```

对于 `patch_len=8`：

```text
8 days × 5 features = 40 values
        ↓
Linear(40, d_model)
        ↓
每只股票仍为 2 个 patch
        ↓
总图节点仍为 474 × 2 = 948
```

这样保持：

- GraphLearner 不变；
- GCN 不变；
- MoE 不变；
- head 不变；
- 股票节点语义不变。

第一版 EOD5 不新增 Feature MLP，只做：

```text
8×5 → Linear → d_model
```

这样可以单独验证“多特征输入”本身的贡献。

实现前必须确认 `SP500.npy` 5 个 channel 的真实语义、排列和尺度，不能未经验证直接假定为 raw OHLCV。

---

### 2. 修正 RankLoss 权重

当前：

```yaml
rank_weight: 0.1
```

最新梯度诊断显示，在 Epoch 5：

```text
||∇(0.1 RankLoss)|| / ||∇MSE|| ≈ 0.0089
```

即 RankLoss 梯度不足 MSE 的 1%，目前几乎没有真正参与优化。

因此不要继续默认沿用 StockMixer 的 `0.1`。

建议优先测试：

```text
rank_weight = 1
rank_weight = 5
rank_weight = 10
```

每组继续使用已有 gradient diagnostic：

```text
epoch 0
epoch 1
epoch 5
```

主要判断：

```text
||∇Rank|| / ||∇MSE||
```

目标不是强制两者完全相等，而是避免 RankLoss 再低两个数量级。

建议优先希望 RankLoss 梯度达到：

```text
约 0.2 ~ 1.0 × MSE gradient
```

---

### 3. 核对具体运行的日期划分和指标口径

当前 TimeFilter 的 SP500 划分已与 StockMixer **论文**一致（1006 / 253 / 352 天），无需重新切分。需要核对的是本地 StockMixer 具体实验的保存配置：`20260705_165640_SP500` 使用了 756 / 1008 边界，与当前 TimeFilter 的测试日期不同；StockMixer 训练循环还需确保打乱的窗口偏移不会让训练目标跨入验证段。

因此当前：

```text
StockMixer 指标
vs
TimeFilter 指标
```

不能直接解释为模型结构差异。

必须统一：

```text
同一 SP500.npy
同一 Train 日期
同一 Validation 日期
同一 Test 日期
同一 seq_len = 16
同一 t → t+1 标签定义
同一评价指标
```

统一后才能进行：

```text
StockMixer vs TimeFilter
Return1 vs EOD5
```

的公平比较。

---

## P1：高优先级

### 4. 解决 Norm OFF 后的输入尺度问题

当前：

```yaml
norm: false
```

避免了 TimeFilter 原始 Normalize 最后的历史均值恢复问题。

但日收益率通常只有：

```text
O(1e-2)
```

而 positional embedding 大约是：

```text
O(1)
```

EOD5 的不同 feature 之间还可能存在更严重的尺度差异。

因此建议增加金融专用：

```text
scale-only normalization
```

形式：

```text
z = x / (scale + eps)
prediction = prediction_normalized × scale
```

不要恢复历史均值：

```text
不要 + mean
```

目标：

```text
避免 historical mean shortcut
+
让输入尺度接近 O(1)
```

如果 EOD5 数据本身已经是 StockMixer 预处理后的统一尺度，则优先保持原始数据定义，不额外重复标准化。

任何统计量只能使用 Train period 计算。

---

### 5. 降低模型容量

当前：

```yaml
d_model: 512
d_ff: 2048
e_layers: 2
```

对于约 990 个高度重叠训练窗口仍然偏大。

建议固定前面的输入、loss 和 split 后，再比较：

```yaml
# Candidate A
d_model: 256
d_ff: 1024
e_layers: 1
```

```yaml
# Candidate B
d_model: 128
d_ff: 512
e_layers: 1
```

原：

```yaml
512 / 2048 / 2
```

作为对照保留。

重点看：

```text
Validation RankIC
Validation IC
Train-Val gap
Test 泛化
```

---

### 6. `pos=1` vs `pos=0` 消融

当前 positional embedding 实际作用在：

```text
Stock0-patch0
Stock0-patch1
Stock1-patch0
Stock1-patch1
...
```

股票编号本身没有天然线性顺序，因此：

```text
stock index positional encoding
```

语义可疑。

建议做：

```text
pos = 1
vs
pos = 0
```

其他配置完全不变。

尤其在 `norm=false` 时，PE 数值可能比原始收益率信号更强，因此该实验优先级较高。

---

### 7. 修正图 mask 后 softmax 的“伪稀疏”问题

当前逻辑：

```text
adj = adj * mask
        ↓
被删除边变成 0
        ↓
softmax(adj)
```

由于：

```text
exp(0) = 1
```

被 mask 掉的边在 softmax 后仍重新获得非零权重。

因此当前：

```text
KNN pruning
MoE relation mask
alpha
```

都不是真正严格意义上的 sparse graph。

后续应改为严格 mask，例如：

```text
被删除位置 → -inf
再做 softmax
```

或使用其他保持稀疏性的归一化方式。

这一步会改变图传播逻辑，因此应在输入、loss、normalization、capacity 稳定后单独验证。

---

## P2：结构级问题

### 8. MoE gate 与预测目标脱节

当前梯度诊断已经确认：

```text
MSE → MoE gate gradient = 0
RankLoss → MoE gate gradient = 0
```

MoE gate 只由 auxiliary loss 训练。

这意味着 routing 学到的是：

```text
expert 分布 / 均衡性
```

而不是：

```text
哪类关系最有利于下一日股票预测
```

这是明确的结构性瓶颈。

但该问题涉及：

```text
hard routing
可微 routing
Gumbel / STE
finance-aware gating
```

属于结构改造，不建议在基础金融 baseline 尚未稳定前优先修改。

---

### 9. alpha / patch 暂时后置

当前建议暂时固定：

```yaml
patch_len: 8
alpha: 0.3
```

原因：

- `patch=8` 已经恢复 Temporal / ST relation；
- `alpha=0.3` 比 `0.7` 更温和；
- 当前更大的问题已经明确集中在输入、RankLoss、scale 和泛化。

暂时不优先做：

```text
alpha = 0.25 / 0.30 / 0.35
patch = 4 / 8
```

这种细粒度搜索。

---

# 推荐执行顺序

```text
① 核对 StockMixer / TimeFilter 的实际数据、目标日、窗口、测试日及评估口径
        ↓
② 增加 Return1 / EOD5 输入切换
        ↓
③ 重调 RankLoss 权重并重新做梯度诊断
        ↓
④ 引入/验证 scale-only normalization
        ↓
⑤ 降低模型容量
        ↓
⑥ pos=0 / pos=1 消融
        ↓
⑦ 修正严格图 mask / softmax
        ↓
⑧ 再处理 MoE routing 与预测目标脱节
        ↓
⑨ 最后进入 Signed Graph
```

---

# 当前核心判断

目前最优先解决的不是继续改图结构，而是：

```text
1. 输入是否公平
2. RankLoss 是否真的在训练模型
3. 具体运行的数据划分和评估口径是否统一
```

这三项没有解决之前：

```text
TimeFilter vs StockMixer
```

的差距不能直接归因于模型架构。

同时，Signed Graph 应放在这些基础问题解决之后，否则后续性能提升很难明确归因。
