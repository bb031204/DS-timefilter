# TimeFilter EOD5 五特征实验：当前核心问题概述

> 背景：当前五特征实验将 S&P500 的 5 维 EOD 行情输入 TimeFilter，希望在保持原有图结构的基础上提升横截面股票预测能力。实验结果显示，EOD5 相比 Return1 没有带来提升，尤其 RankIC、Sharpe 等排序/选股指标明显退化。当前更需要排查的是五特征在 TimeFilter 中的表示方式、目标接口和优化方式，而不是直接认定“五特征无效”。

---

## P0-1：RankLoss 在 TimeFilter 中几乎没有实际优化作用

### 现象

当前训练目标为：

```text
Loss = MSE + 0.1 × RankLoss + 0.005 × MoE Loss
```

最新 EOD5 梯度诊断显示，在 Epoch 5：

```text
||∇(0.1 × RankLoss)|| / ||∇MSE|| ≈ 0.006
```

也就是 RankLoss 的梯度规模仅约为 MSE 的 **0.6%**。

### 为什么可能导致性能不提升

当前任务的核心目标不仅是降低收益率点预测误差，还需要提升股票之间的横截面排序能力，例如 IC、RankIC、Precision@10 和 Sharpe。

当 RankLoss 梯度远小于 MSE 时，模型训练实际上近似：

```text
Loss ≈ MSE
```

模型会优先学习降低整体数值误差，而不是学习：

```text
Stock_i > Stock_j ?
```

因此即使 EOD5 提供了更多信息，模型也可能没有足够的优化压力把这些信息转化为有效的排序信号。

### 当前证据等级

**直接证据，优先级最高。**

### 后续验证方向

优先对：

```text
rank_weight = 1 / 5 / 10
```

做梯度诊断，而不是直接全部训练 100 Epoch。

重点观察：

```text
||∇Rank|| / ||∇MSE||
```

建议先把 RankLoss 梯度提高到大约：

```text
0.1 ~ 0.5 × MSE gradient
```

再进行完整训练。

---

## P0-2：EOD5 直接预测 Return，与 StockMixer 的 Price→Return 接口不同

### 现象

当前 TimeFilter EOD5 的流程是：

```text
过去 16 天五维 EOD
        ↓
TimeFilter
        ↓
直接预测下一日 return
```

即：

\[
X_{EOD} \rightarrow \hat r_{t+1}
\]

而 StockMixer 的原始流程更接近：

```text
过去 16 天五维 EOD
        ↓
StockMixer
        ↓
预测下一日价格
        ↓
与当前价格比较
        ↓
转换为下一日 return
```

即：

\[
X_{EOD} \rightarrow \hat P_{t+1}
\rightarrow
\hat r_{t+1}
=
\frac{\hat P_{t+1}-P_t}{P_t}
\]

### 为什么可能导致性能不提升

EOD5 中的大量信息反映的是价格水平、价格形态和成交量状态，而下一日收益率通常是一个很小的相对变化量。

当前 TimeFilter 要直接学习：

```text
EOD 状态 → 微小 return
```

需要模型自己隐式学习“相对当前价格变化”的关系。

StockMixer 则显式使用当前价格作为锚点，将预测价格转换为收益率，因此任务接口与五维行情输入更匹配。

### 当前证据等级

**高度可疑，但尚未完成直接消融验证。**

### 后续验证方向

设计单独对照：

```text
A. EOD5 → direct return
B. EOD5 → next price → return
```

其它条件保持一致。

---

## P0-3：EOD5 输入没有 Normalization，而 StockMixer 内部有 LayerNorm

### 现象

当前 TimeFilter EOD5 路径为：

```text
EOD5
→ reshape / patch
→ Linear
→ Graph Backbone
```

并且：

```yaml
norm: false
```

因此 EOD5 不经过原 TimeFilter 的 Normalize，也没有其它输入端 normalization。

而 StockMixer 内部包含 LayerNorm，用于在 mixing 前稳定输入和中间表示尺度。

### 为什么可能导致性能不提升

五个 EOD feature 的数值范围、方差和分布可能不同。

直接执行：

```text
8 × 5 → Linear
```

时，大尺度或高方差特征可能主导 patch projection。

同时 TimeFilter 还加入 positional embedding，如果输入投影本身尺度不稳定，会进一步影响表示质量。

因此可能出现：

```text
输入信息更多
但有效表示反而更差
```

### 当前证据等级

**高度可疑。**

目前代码已经确认 EOD5 完全绕过输入 normalization，但还缺少五个 feature 的实际统计量诊断。

### 后续验证方向

先统计 Train 段五个 feature 的：

```text
mean
std
min
max
quantiles
```

然后优先尝试“仅输入端”的 normalization，而不是恢复旧 RevIN 行为。

推荐比较：

```text
A. EOD5 raw
B. EOD5 + input LayerNorm
C. EOD5 + train-stat scale-only normalization
```

注意：

```text
只归一化输入
不要对 return 输出恢复历史均值
```

任何统计量只能从训练集计算。

---

## P1-1：`8×5 → Linear` 没有显式 Indicator Modeling

### 现象

当前每只股票的一个 patch 使用：

```text
8 天 × 5 特征
        ↓
展平为 40 维
        ↓
Linear(40, d_model)
```

这能保持每只股票仍有 2 个 patch、总图节点仍为 948，因此作为第一版输入消融是合理的。

但它只是把时间维和特征维一起压缩。

### 为什么可能导致性能不提升

StockMixer 使用的不是简单的“五特征输入”，而是专门设计了：

```text
Indicator Mixing
Temporal Mixing
Stock Mixing
```

分别处理指标间关系、时间关系和股票关系。

当前 TimeFilter 的：

```text
8×5 → Linear
```

没有显式区分：

```text
哪一维是时间
哪一维是 indicator
```

因此新增的五维信息可能在进入 Graph Backbone 前就被过度压缩。

也就是说：

```text
更多特征 ≠ 有效的多特征建模
```

当前实验只能说明：

```text
naive EOD5 patch projection 没有带来提升
```

不能说明：

```text
五特征对 TimeFilter 无效
```

### 当前证据等级

**较高可能性，但仍属于结构假设，需要消融验证。**

### 后续验证方向

在 RankLoss 和 normalization 问题解决后，再测试轻量级 indicator encoder，例如：

```text
5 features
→ small feature projection / feature mixer
→ temporal patch
→ TimeFilter backbone
```

第一阶段不建议直接引入复杂 Indicator Mixer，否则很难区分性能变化来自输入、normalization 还是新增结构。

---

# 四个问题之间的关系

当前 EOD5 的实际流程可以概括为：

```text
五维 EOD
    ↓
无输入 normalization
    ↓
8×5 直接 Linear
    ↓
TimeFilter Graph Backbone
    ↓
直接预测 return
    ↓
训练目标主要由 MSE 主导
RankLoss 梯度仅约 MSE 的 0.6%
```

因此当前实验无提升，更可能是以下因素共同作用：

```text
输入表示不充分
+
尺度处理不足
+
预测目标接口不匹配
+
排序目标优化太弱
```

而不是简单的：

```text
“五特征没有用”
```

---

# 当前建议的排查顺序

```text
P0-1 先修正 RankLoss 有效梯度
        ↓
P0-3 检查并处理 EOD5 输入尺度
        ↓
P0-2 对比 direct-return 与 price→return 接口
        ↓
P1-1 再考虑显式 indicator modeling
```

在这些问题确认之前，不建议继续增加更多特征、修改复杂图结构或直接引入 Signed Graph。

---

# 当前实验应如何定性

建议将本次结果表述为：

> 当前实验表明，简单地将五维 EOD 行情以 `8×5→Linear` 的方式接入 TimeFilter，并在现有 direct-return、无输入 normalization、低 RankLoss 权重的训练配置下，并不能提升模型性能，且横截面排序指标明显下降。该结果不能说明五特征本身无效，更可能表明 TimeFilter 仍缺少与多维金融输入相匹配的表示、归一化和优化机制。
