# DegFiLM 全面拆解：一组自适应的"旋钮"

> 于陈远 | 南京信息工程大学 22级信安1班 | 202283290014  
> 日期：2026.5.20  
> 对应模型：`models/deg_film_blocks.py`、`models/degfilm_resunet.py`  
> 前置阅读：`DegEstimator_全面拆解.md`

---

## 一、引言：从"固定药方"到"对症下药"

如果说 DegEstimator 是给网络加的"眼睛"，那 **DegFiLM 就是给网络装的一组"旋钮"**。

A方案（PureResUNet）的核心问题是**修复策略固定**：无论输入帧的压缩强度如何，所有卷积核的权重都是训练好后固定不变的。QP22（轻微伪影）和 QP42（严重块效应）被同一套参数处理，这 inevitably 导致"小病猛治"或"大病轻治"。

DegFiLM 的设计目标很简单：**让网络的修复力度，根据输入帧的退化程度动态可调**。它由两个紧密耦合的组件构成：

1. **FiLM 生成器**：从 DegEstimator 输出的退化嵌入，生成逐通道的缩放因子 γ 和平移量 β。
2. **FiLMResBlock**：在标准残差块内部注入 `(1+γ)·body_out + β`，实现条件化的残差调节。

本文将逐层拆解 DegFiLM 的数学原理、实现细节、训练动态和设计权衡。

---

## 二、FiLM 生成器：从退化嵌入到控制信号

### 2.1 代码实现

```python
class FiLM(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, embed):
        return self.fc(embed)  # (B, 2*out_channels)
```

### 2.2 输入输出

- **输入**：`embed` (B, 64)，来自 DegEstimator 的退化嵌入
- **输出**：`film_params` (B, 2*C)，前 C 维为 γ，后 C 维为 β
- **参数**：`embed_dim × 2C + 2C`

以 Bottleneck 层为例（C=256）：
- `self.fc = nn.Linear(64, 512)`
- 参数量：64 × 512 + 512 = **33,280**

### 2.3 为什么只用一个 Linear 层？

FiLM 生成器的设计遵循**极简主义**：一个 Linear 层，无激活函数，无 BN，无隐藏层。

**理由**：
1. **退化嵌入已经经过高度非线性变换**：DegEstimator 的 3 层 CNN + ReLU 已经把原始像素映射到了高维非线性空间。FiLM 生成器不需要再做复杂的特征提取，它只需要做一个**投影**——把 64 维嵌入空间映射到 2C 维的控制空间。
2. **避免梯度消失/爆炸**：多层 MLP 会增加梯度传播路径长度，而 FiLM 的梯度需要回传到 DegEstimator。保持单层可以减少不稳定因素。
3. **保持轻量**：每个 FiLM 层只有 33K 参数，8 个 FiLM 层总计 0.18M，占 B 方案总参数的 **1.47%**。

### 2.4 初始化策略：全零初始化

```python
nn.init.zeros_(self.fc.weight)
nn.init.zeros_(self.fc.bias)
```

这是 DegFiLM 设计中最关键的细节之一。全零初始化意味着：**在训练初期，无论 DegEstimator 输出什么嵌入，FiLM 层都输出 γ=0, β=0**。

我们将在第 6 章深入分析这个设计的精妙之处。

---

## 三、FiLMResBlock：条件化残差块

### 3.1 代码实现

```python
class FiLMResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x, film_params=None):
        out = self.body(x)
        if film_params is not None:
            B, C = out.size(0), out.size(1)
            gamma = film_params[:, :C].view(B, C, 1, 1)
            beta  = film_params[:, C:].view(B, C, 1, 1)
            out = (1 + gamma) * out + beta
        return x + out
```

### 3.2 结构对比：标准 ResBlock vs FiLMResBlock

```
标准 ResBlock (A方案):
    x ──→ [body] ──→ body_out ──→ x + body_out ──→ output

FiLMResBlock (B方案):
    x ──→ [body] ──→ body_out ──→ (1+γ)·body_out + β ──→ x + modulated_out ──→ output
                              ↑
                         film_params from DegEstimator
```

### 3.3 数学形式详解

FiLMResBlock 的前向传播可以写成：

```
output = x + (1 + γ) · body_out + β
```

其中：
- `x`：输入特征图 (B, C, H, W)
- `body_out`：残差分支的输出 (B, C, H, W)
- `γ`：逐通道缩放因子 (B, C, 1, 1)
- `β`：逐通道平移量 (B, C, 1, 1)
- `·`：逐元素乘法（broadcasting 到 H×W）

**关键设计：`(1 + γ)` 而非 `γ`**

这个 `+1` 是 FiLMResBlock 与文献中标准 FiLM 的最大区别。

| 形式 | γ=0 时的行为 | 训练初期效果 | 问题 |
|------|-------------|-------------|------|
| `x + γ·body_out + β` | `x + 0 = x` | 残差通路被**完全关闭** | 训练初期主干网络学不到残差信息 |
| `x + (1+γ)·body_out + β` | `x + body_out` | **退化为标准 ResBlock** | ✅ 正确，不干扰主干学习 |

**物理意义**：
- `γ > 0`：放大残差出力，增强修复力度（适用于重度压缩）。
- `γ < 0`：缩小残差出力，减弱修复力度（适用于轻度压缩，避免过度平滑）。
- `γ = 0`：标准残差块，不增不减。
- `β`：调整特征分布的均值，防止 `(1+γ)` 缩放后分布漂移，避开 ReLU 死亡区。

### 3.4 为什么 β 也很重要？

假设只有 γ 没有 β：

```
output = x + (1 + γ) · body_out
```

当 γ 为很大的正值时，`body_out` 被大幅放大，可能导致特征值分布整体偏移，后续 BatchNorm 层需要花很多 epoch 重新适应。β 的存在让 FiLM 可以同时调节**尺度**和**位置**，提供更灵活的控制。

### 3.5 广播机制：从 (B, C) 到 (B, C, H, W)

```python
gamma = film_params[:, :C].view(B, C, 1, 1)  # (B, C, 1, 1)
beta  = film_params[:, C:].view(B, C, 1, 1)  # (B, C, 1, 1)
out = (1 + gamma) * out + beta               # broadcasting over H, W
```

注意：γ 和 β 是**逐通道**的，同一通道内的所有空间位置共享同一个 γ 值和 β 值。这意味着 FiLM 的调节粒度是**通道级**的，而非**空间级**的。

- **优点**：计算高效，参数量少，不容易过拟合。
- **缺点**：无法做到"左上角多修、右下角少修"的空间自适应。这是当前设计的一个局限（见第 9 章）。

---

## 四、DegFiLMResUNet：整体装配

### 4.1 插入位置总览

| 层级 | 块数 | 是否 FiLM | 通道数 C | 每块 FiLM 参数量 | 该层 FiLM 总参数量 |
|------|------|----------|---------|----------------|------------------|
| Encoder Level 0~3 | 2×4 | ❌ 不插 | — | — | — |
| **Bottleneck** | **2** | ✅ | 256 | 33,280 | **66,560** |
| **Decoder Level 0** | **2** | ✅ | 256 | 33,280 | **66,560** |
| **Decoder Level 1** | **2** | ✅ | 128 | 16,640 | **33,280** |
| **Decoder Level 2** | **2** | ✅ | 64 | 8,320 | **16,640** |
| Decoder Level 3 | 2 | ❌ 不插 | — | — | — |
| **FiLM 总计** | **8** | | | | **183,040** |

加上 DegEstimator 的 101,728，B 方案总增量：**284,768** (~0.285M)。

对比基线：
- A 方案 PureResUNet：12,158,144 (~12.16M)
- B 方案 DegFiLMResUNet：12,442,912 (~12.44M)
- **增量比例：2.34%**

### 4.2 前向传播完整流程

```python
def forward(self, x):
    identity = x
    x = self.init_conv(x)

    # ===== Encoder（标准 ResBlock，不插 FiLM）=====
    skips = []
    for i in range(4):
        for res in self.enc_blocks[i]:
            x = res(x)           # 标准 ResBlock
        skips.append(x)
        x = self.down_blocks[i](x)

    # ===== DegEstimator（只算一次，供深层复用）=====
    deg_embed = self.deg_estimator(identity)  # (B, 64)

    # ===== Bottleneck（FiLMResBlock × 2）=====
    for idx, res in enumerate(self.bottleneck):
        film_params = self.film_bottleneck[idx](deg_embed)  # (B, 512)
        x = res(x, film_params)

    # ===== Decoder（前3层 FiLMResBlock，最浅层标准 ResBlock）=====
    for i in range(4):
        x = self.up_blocks[i](x)
        skip = skips[3 - i]

        if i < 3:
            # level 0,1,2：手动处理 cat + conv，然后注入 FiLM
            x = self.dec_blocks[i][0](torch.cat([x, skip], dim=1))
            for j in range(2):
                film_params = self.film_decoder[i][j](deg_embed)
                x = self.dec_blocks[i][j + 1](x, film_params)
        else:
            # level 3（最浅层）：标准 _fuse_skip，不插 FiLM
            x = self._fuse_skip(x, skip, i)

    x = self.out_conv(x)
    return x + identity
```

### 4.3 为什么 Encoder 不插 FiLM？

这是 B 方案最核心的结构设计之一，有三层理由：

**理由1：功能分离**
- Encoder 的职责是**提取内容特征**（边缘、纹理、颜色、语义）。
- 如果在 Encoder 中插入 FiLM，退化信息会在网络最浅层就介入内容表征的学习，可能导致"内容特征被退化信息污染"。
- 类比：画画时，素描阶段（Encoder）应该专注于轮廓和结构，上色阶段（Decoder）才需要根据画面整体氛围调整色调。

**理由2：PromptCIR 的启发**
- PromptCIR 原文明确指出 Prompt（条件信息）应该在 **Decoder 阶段**与图像特征交互，而不是 Encoder。
- Encoder 负责"理解内容"，Decoder 负责"生成/修复内容"。条件化应该作用于"生成"阶段。

**理由3：梯度稳定性**
- Encoder 靠近输入，梯度路径长。如果在 Encoder 中引入 FiLM，DegEstimator 的梯度需要经过更多层才能回传，增加不稳定风险。
- Bottleneck 和 Decoder 靠近输出，梯度信号更直接、更清晰。

### 4.4 为什么 Decoder 最浅层（Level 3）不插 FiLM？

Decoder Level 3 的特征图分辨率最高（与输入相同，如 1920×1080），通道数最少（32）。

- **分辨率太高**：如果插入 FiLM，γ 和 β 的广播乘法需要在全分辨率上进行，计算开销显著增加。
- **通道数太少**：32 通道的 FiLM 控制空间很有限，调节收益小。
- **职责定位**：最浅层主要负责细节精修（如锐化、去噪），这些操作对"全局退化强度"不敏感，不需要条件化。

### 4.5 为什么 Bottleneck 要插 FiLM？

Bottleneck 是网络的最深层，特征图分辨率最低（1/16），通道数最高（256），感受野最大。

- **感受野最大**：Bottleneck 能看到整帧图像的上下文，最适合做"全局决策"——比如"这帧整体压缩很严重，需要大力修复"。
- **信息瓶颈**：Bottleneck 是 Encoder 和 Decoder 的桥梁，在这里注入退化信息，可以让后续所有 Decoder 层都受益。
- **通道数最多**：256 通道提供了充足的 FiLM 控制自由度。

---

## 五、数学推导：FiLM 的梯度流

### 5.1 前向传播的数学表达式

设 FiLMResBlock 的输入为 `x`，输出为 `y`：

```
body_out = Body(x)
y = x + (1 + γ) · body_out + β
```

其中 `Body(·)` 表示残差分支（BN-ReLU-Conv-BN-ReLU-Conv）。

### 5.2 反向传播：FiLM 参数如何更新？

假设损失函数为 `L`，我们需要求 `∂L/∂γ` 和 `∂L/∂β`。

```
∂L/∂γ = ∂L/∂y · ∂y/∂γ = ∂L/∂y · body_out
∂L/∂β = ∂L/∂y · ∂y/∂β = ∂L/∂y · 1
```

由于 `γ` 和 `β` 是通过 FiLM 生成器的 Linear 层从 `deg_embed` 生成的：

```
γ = W_γ · deg_embed + b_γ
β = W_β · deg_embed + b_β
```

所以：

```
∂L/∂deg_embed = (∂L/∂γ · W_γ) + (∂L/∂β · W_β)
```

这个梯度会进一步回传到 DegEstimator，告诉它："你应该输出什么样的退化嵌入，才能让 FiLM 的调节更有利于降低损失。"

### 5.3 γ 的梯度与 body_out 的关系

注意 `∂L/∂γ = ∂L/∂y · body_out`。这意味着：

- 如果 `body_out` 很大（残差分支输出强），γ 的梯度也会很大，FiLM 层可以快速学会"何时该增强"。
- 如果 `body_out` 很小（残差分支输出弱），γ 的梯度也小，FiLM 层学会"保持默认"。

这种**自适应梯度幅度**让 FiLM 的训练非常稳定——它不会盲目地大幅调整 γ，而是根据残差分支的实际出力来决定调节力度。

---

## 六、初始化策略的深度分析

### 6.1 全零初始化的精妙之处

FiLM 生成器的权重和偏置都初始化为0：

```python
nn.init.zeros_(self.fc.weight)
nn.init.zeros_(self.fc.bias)
```

这在训练初期导致：

```
γ = W_γ · deg_embed + b_γ = 0 · deg_embed + 0 = 0
β = W_β · deg_embed + b_β = 0 · deg_embed + 0 = 0
```

代入 FiLMResBlock：

```
output = x + (1 + 0) · body_out + 0 = x + body_out
```

**这就是标准 ResBlock 的行为。**

### 6.2 为什么这很重要？

如果没有全零初始化，假设权重随机初始化（如 Xavier）：

```
γ ~ Uniform(-a, a)，其中 a 可能达到 0.1~1.0
```

训练初期：
- 某些通道的 γ = +0.5 → 残差出力被放大 50%
- 某些通道的 γ = -0.5 → 残差出力被缩小 50%
- 这会导致特征分布剧烈震荡，主干网络需要同时学习"如何修复"和"如何适应 FiLM 的随机扰动"

**全零初始化实现了"渐进式引入"**：
1. Epoch 0~10：FiLM ≈ 不存在，网络先学会 A 方案的基本修复能力。
2. Epoch 10~30：FiLM 开始微调，在已有基础上做小幅优化。
3. Epoch 30+：FiLM 完全激活，网络实现退化感知修复。

这种**课程学习式**的激活策略，比一开始就全开要稳定得多。

### 6.3 与 ResNet 恒等映射的类比

ResNet 的成功很大程度上归功于恒等映射（Identity Mapping）：`output = x + F(x)`。当 `F(x)` 难以学习时，网络可以退化到 `output ≈ x`，保证训练不会崩。

FiLMResBlock 的全零初始化继承了同样的哲学：
- **ResNet**：通过残差连接保证"至少不会更差"。
- **FiLMResBlock**：通过全零初始化保证"FiLM 至少不会干扰已有的残差学习"。

---

## 七、与文献的精确对应

### 7.1 vs FiLM (AAAI 2018)

FiLM 原文提出的通用形式是：

```
FiLM(F | γ, β) = γ · F + β
```

我们的实现做了两处本地化适配：

| 原文 | 本方案 | 理由 |
|------|--------|------|
| `γ · F + β` | `(1+γ) · body_out + β` | 保证 γ=0 时退化为标准 ResBlock |
| 条件输入任意 | 条件输入 = DegEstimator 退化嵌入 | 针对视频压缩伪影去除任务定制 |
| 插入位置任意 | 只插 Bottleneck + Decoder | 遵循功能分离原则 |

### 7.2 vs PromptCIR (CVPRW 2024)

PromptCIR 的核心是"轻量 Prompt 与图像特征动态交互"。

| PromptCIR | 本方案 | 对应关系 |
|-----------|--------|---------|
| Prompt 生成器 | DegEstimator | 都是轻量旁路，提取条件信息 |
| Prompt 与特征交互 | FiLM 注入 | Prompt 调节 Soft Weights，FiLM 调节通道统计量 |
| Decoder 导向 | Encoder 不插 | 条件化只作用于重建阶段 |
| 端到端 | 端到端 | 都不需要预训练 |

**核心差异**：PromptCIR 的交互更复杂（涉及空间注意力），我们的 FiLM 更简单（只调通道统计量），但代价也更低。

---

## 八、消融实验设计

以下是围绕 DegFiLM 可以做的系统性消融：

### 8.1 消融1：FiLM 生成器深度

- **变体A（当前）**：1 层 Linear
- **变体B**：2 层 MLP（Linear-ReLU-Linear）
- **变体C**：0 层（直接把 deg_embed 复制成 2C 维，不做任何变换）

**预期**：变体A 最优。变体B 可能过拟合，变体C 可能表达能力不足。

### 8.2 消融2：`(1+γ)` vs `γ`

- **变体A（当前）**：`(1+γ) · body_out + β`
- **变体B**：`γ · body_out + β`

**预期**：变体A 训练更稳定，最终性能更好。变体B 初期训练可能震荡。

### 8.3 消融3：只保留 γ，去掉 β

- **变体A（当前）**：有 γ 有 β
- **变体B**：只有 γ，β 固定为 0

**预期**：变体A 更好，β 对分布稳定性有贡献。

### 8.4 消融4：FiLM 插入密度

- **变体A（当前）**：Bottleneck 2 + Decoder 6 = 8 个 FiLMResBlock
- **变体B**：全部插入（Encoder 4 + Bottleneck 2 + Decoder 8 = 14 个）
- **变体C**：只插 Bottleneck（2 个）
- **变体D**：只插 Decoder（6 个）

**预期**：变体A 最优。变体B 可能让 Encoder 特征被污染；变体C/ D 条件化不足。

### 8.5 消融5：γ 的可视化分析

统计不同 QP 下各层 γ 的分布：

```python
# 伪代码
for qp in [22, 27, 32, 37, 42]:
    gamma_means = []
    for layer in film_layers:
        gamma = layer(deg_est(input_qp))
        gamma_means.append(gamma.mean().item())
    print(f"QP{qp}: γ均值分布 = {gamma_means}")
```

**预期**：QP42 的 γ 均值 > QP22 的 γ 均值，且 Bottleneck 的 γ 幅度 > Decoder 浅层。

---

## 九、局限性与改进方向

### 9.1 局限1：通道级调节，无空间选择性

当前 FiLM 的调节粒度是 `(B, C, 1, 1)`，同一通道的所有空间位置被同等缩放。这意味着：

- ✅ "这一帧的纹理通道需要多修"
- ❌ "这一帧的左上角需要多修，右下角少修"

**改进方向：Spatial-FiLM**

让 FiLM 生成器输出空间特征图 `(B, C, H, W)` 而非向量 `(B, C, 1, 1)`：

```python
# 改进版 FiLM 生成器
self.fc = nn.Linear(embed_dim, C * 2 * H * W)  # 输出完整空间图
# 或者
self.conv = nn.Conv2d(embed_dim, C * 2, 3, padding=1)  # 保持空间结构
```

代价：参数量和计算量大幅增加。需要权衡。

### 9.2 局限2：单帧退化感知，无时域一致性

DegEstimator 只看当前帧。如果相邻帧的 QP 相同但内容运动剧烈，DegEstimator 可能输出不一致的嵌入，导致时域闪烁。

**改进方向：时域 DegEstimator**

输入当前帧 + 前后各1帧，用 3D 卷积或时域注意力提取时域一致的退化特征。

### 9.3 局限3：γ 范围无约束

当前 γ 是自由学习的，理论上可以趋于 ±∞。虽然实践中梯度裁剪和权重衰减会约束它，但如果某个通道的 γ 学得过大，可能导致特征爆炸。

**改进方向：门控 FiLM**

```python
# 把 γ 限制在 [-1, 1] 或 [0, 1]
gamma = torch.tanh(gamma)  # [-1, 1]，残差出力范围 [0, 2]
# 或
gamma = torch.sigmoid(gamma)  # [0, 1]，残差出力范围 [1, 2]
```

### 9.4 局限4：单一退化嵌入驱动所有 FiLM 层

DegEstimator 只输出一个 64 维向量，供给所有 8 个 FiLM 层使用。这假设了"一帧图像只需要一种退化描述"，且"所有层需要同样的退化信息"。

**改进方向：分层退化嵌入**

让 DegEstimator 输出多尺度嵌入，不同 FiLM 层接收不同级别的退化信息：

```python
# 改进版 DegEstimator
deg_embeds = {
    'bottleneck': self.fc_bottleneck(feat),      # 64-dim
    'decoder_l0': self.fc_decoder_l0(feat),      # 64-dim
    'decoder_l1': self.fc_decoder_l1(feat),      # 64-dim
    # ...
}
```

---

## 十、总结

DegFiLM 是 B 方案的核心创新，它用极低的代价（**0.18M 参数，1.47% 增量**）实现了网络修复策略的动态条件化。

| 组件 | 职责 | 关键设计 |
|------|------|---------|
| **FiLM 生成器** | 把退化嵌入翻译为控制信号 | 单层 Linear，全零初始化 |
| **FiLMResBlock** | 根据控制信号调节残差出力 | `(1+γ)·body_out + β`，退化为标准 ResBlock |
| **分层插入** | 在合适的位置发挥最大效用 | Encoder 不插，Bottleneck+Decoder 插，最浅层不插 |

**从测试结果看**，DegFiLM 在 QP22~37 展现了微弱但一致的相对优势（~10%），但在 QP42 未能突破 A 方案的瓶颈。这提示我们：

> **通道级 FiLM 的调节粒度，对于重度压缩场景可能 insufficient。** 未来的改进方向应该是从"通道级"走向"空间-通道联合级"，或者引入时域一致性约束。

但无论如何，DegFiLM 的设计验证了一个重要的方法论：**在已有强基线的基础上，最小侵入式地添加条件化能力，是探索新方向的安全且高效的方式。**

---

## 附录：快速参考

### FiLM 参数量速查表

| 目标通道 C | Linear 维度 | 每 FiLM 层参数量 |
|-----------|------------|----------------|
| 32 | 64 → 64 | 4,160 |
| 64 | 64 → 128 | 8,320 |
| 128 | 64 → 256 | 16,640 |
| 256 | 64 → 512 | 33,280 |
| 512 | 64 → 1024 | 66,048 |

### 代码速览

```python
from models.deg_film_blocks import FiLM, FiLMResBlock

# FiLM 生成器
film_gen = FiLM(embed_dim=64, out_channels=256)
embed = torch.randn(2, 64)       # (B, 64) 退化嵌入
film_params = film_gen(embed)     # (B, 512)，前256=γ，后256=β

# FiLMResBlock
block = FiLMResBlock(channels=256)
x = torch.randn(2, 256, 32, 32)  # (B, C, H, W)
out = block(x, film_params)       # (B, 256, 32, 32)
```

### B方案增量总结

| 模块 | 参数量 | 占比 |
|------|--------|------|
| A 方案 PureResUNet | 12,158,144 | 基准 |
| DegEstimator | 101,728 | 0.82% |
| FiLM 相关（8层） | 183,040 | 1.47% |
| **B 方案总增量** | **284,768** | **2.34%** |
| **B 方案总计** | **12,442,912** | **100%** |
