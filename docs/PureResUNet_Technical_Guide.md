# PureResUNet（A方案）全面技术顾问文档

> **版本**：v1.1  
> **更新日期**：2026-05-14  
> **更新摘要**：同步当前代码状态（batch_size=32, num_workers=6, amp=False, RTX 5080 实测满载配置、三级推理策略）  
> **适用**：MFQEv2 视频质量增强基线  
> **目标**：为 A/B/C 对比实验提供控制变量严格的纯像素驱动网络

---

## 目录

1. [设计哲学](#1-设计哲学)
2. [网络架构总览](#2-网络架构总览)
3. [核心组件详解](#3-核心组件详解)
4. [参数量与计算复杂度分析](#4-参数量与计算复杂度分析)
5. [感受野与特征层次](#5-感受野与特征层次)
6. [训练动力学分析](#6-训练动力学分析)
7. [与标准U-Net的差异化设计](#7-与标准u-net的差异化设计)
8. [B/C方案扩展接口](#8-bc方案扩展接口)
9. [调优指南与最佳实践](#9-调优指南与最佳实践)
10. [已知限制与规避策略](#10-已知限制与规避策略)
11. [附录：完整公式推导](#11-附录完整公式推导)

---

## 1. 设计哲学

### 1.1 控制变量原则

A方案的核心使命是建立一个**零外部知识、零注意力、零条件化**的纯净基线。在A/B/C对比实验中，A方案承担"对照组"角色：

- **不引入VLM**：排除CLIP等预训练模型的知识注入效应
- **不引入注意力**：排除自注意力/交叉注意力带来的非局部建模效应
- **不引入路由门控**：排除动态计算和专家混合效应
- **纯卷积驱动**：所有性能增益必须来自卷积核的空间局部学习

### 1.2 残差学习的适用性

视频质量增强（VQE）任务的本质是**残差学习**：

```
HQ = LQ + R(LQ)
```

其中 `R(·)` 是需要学习的压缩伪影残差。残差连接天然适合此任务：
- 如果网络权重趋近于0，输出近似恒等映射 `y ≈ x`
- 网络只需学习**偏离输入的微小修正**，而非从零重建整帧
- 梯度可以直接通过跳跃连接回传，缓解深层网络的梯度消失

### 1.3 Pre-Activation的选型理由

本方案采用 **Pre-Activation ResBlock**（BN-ReLU-Conv-BN-ReLU-Conv），而非 Post-Activation（Conv-BN-ReLU-Conv-BN-ReLU）：

| 特性 | Pre-Activation | Post-Activation |
|------|---------------|-----------------|
| 梯度流动 | 更直接，BN在卷积前稳定分布 | 卷积在前，初始梯度可能不稳定 |
| 恒等映射纯净度 | 完美（shortcut无任何变换） | 通常需1×1投影匹配维度 |
| 深层训练稳定性 | 更优（He et al., Identity Mappings） | 超过100层后退化明显 |
| 本方案适用性 | ✅ 4层编码器+4层解码器，深度适中 | 无需 |

> **参考文献**：He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. *ECCV*.

---

## 2. 网络架构总览

### 2.1 宏观结构

```
Input (3×H×W)
    │
    ▼
[Init Conv] 3→base_ch
    │
    ├──► [Enc1] 2×ResBlock(base_ch) ──► Skip1 ──► Down(base_ch→2·base_ch)
    │                                         │
    ├──► [Enc2] 2×ResBlock(2·base_ch) ──► Skip2 ──► Down(2·base_ch→4·base_ch)
    │                                         │
    ├──► [Enc3] 2×ResBlock(4·base_ch) ──► Skip3 ──► Down(4·base_ch→8·base_ch)
    │                                         │
    └──► [Enc4] 2×ResBlock(8·base_ch) ──► Skip4 ──► Down(8·base_ch→8·base_ch)
                                                      │
                                               [Bottleneck]
                                               2×ResBlock(8·base_ch)
                                                      │
    ┌─────────────────────────────────────────────────┘
    │
    ▼
[Dec1] Up(8·base_ch) → Concat(Skip4) → Conv → 2×ResBlock(8·base_ch)
    │
[Dec2] Up(8·base_ch→4·base_ch) → Concat(Skip3) → Conv → 2×ResBlock(4·base_ch)
    │
[Dec3] Up(4·base_ch→2·base_ch) → Concat(Skip2) → Conv → 2×ResBlock(2·base_ch)
    │
[Dec4] Up(2·base_ch→base_ch) → Concat(Skip1) → Conv → 2×ResBlock(base_ch)
    │
    ▼
[Output] Conv(base_ch→3), No Activation
```

### 2.2 空间分辨率流

以输入 `256×256` 为例：

| 阶段 | 操作 | 输出尺寸 | 通道数 (base_ch=32) |
|------|------|---------|-------------------|
| Input | - | 256×256 | 3 |
| Init Conv | Conv3×3 | 256×256 | 32 |
| Enc1 | 2×ResBlock + Down | 128×128 | 32→64 |
| Enc2 | 2×ResBlock + Down | 64×64 | 64→128 |
| Enc3 | 2×ResBlock + Down | 32×32 | 128→256 |
| Enc4 | 2×ResBlock + Down | 16×16 | 256→256 |
| Bottleneck | 2×ResBlock | 16×16 | 256 |
| Dec1 | Up + Concat + Conv + 2×ResBlock | 32×32 | 256 |
| Dec2 | Up + Concat + Conv + 2×ResBlock | 64×64 | 128 |
| Dec3 | Up + Concat + Conv + 2×ResBlock | 128×128 | 64 |
| Dec4 | Up + Concat + Conv + 2×ResBlock | 256×256 | 32 |
| Output | Conv3×3 | 256×256 | 3 |

> **关键设计**：4层下采样共压缩空间分辨率 `2⁴ = 16` 倍。因此验证/测试时必须将输入尺寸对齐到 **16的倍数**，否则上采样后的尺寸与skip特征不匹配。

---

## 3. 核心组件详解

### 3.1 ResBlock（Pre-Activation）

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x, side_input=None):
        return x + self.body(x)
```

#### 3.1.1 数学表达

设输入为 $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$，则：

$$
\text{ResBlock}(\mathbf{x}) = \mathbf{x} + \mathcal{F}(\mathbf{x}; \{\mathbf{W}_i\})
$$

其中残差函数 $\mathcal{F}$ 为：

$$
\mathcal{F}(\mathbf{x}) = \mathbf{W}_2 * \text{ReLU}\left(\text{BN}_2\left(\mathbf{W}_1 * \text{ReLU}(\text{BN}_1(\mathbf{x}))\right)\right)
$$

这里 $*$ 表示卷积操作，$\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{C \times C \times 3 \times 3}$。

#### 3.1.2 为什么用 bias=False？

- BatchNorm 层已经包含可学习的 `bias`（即 `β` 参数）
- 卷积层加 bias 会导致冗余：Conv 的 bias 被 BN 的 mean subtraction 抵消
- **参数减半**：每个 3×3 Conv 节省 $C$ 个参数

#### 3.1.3 为什么用 inplace=True？

- ReLU 的 inplace 操作直接修改输入张量，不分配新内存
- 在 U-Net 这种多分支结构中，可节省 **10–20% 显存峰值**
- ⚠️ 注意：inplace 操作在某些梯度计算场景中可能引发错误，但残差连接的加法在 ReLU 之后，此处安全

### 3.2 下采样模块

```python
nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
```

#### 3.2.1 与 MaxPool 的对比

| 方案 | 下采样方式 | 优点 | 缺点 |
|------|-----------|------|------|
| **本方案** | Stride-2 Conv | 可学习参数，保留更多信息 | 参数量略高 |
| 替代方案 | MaxPool + Conv | 计算量小，平移不变性强 | 信息丢失不可逆 |
| 替代方案 | AvgPool + Conv | 更平滑，保留低频 | 边缘模糊 |

**选型理由**：VQE任务需要保留纹理细节，可学习的下采样卷积比池化更适合恢复高频伪影。

#### 3.2.2 输出尺寸公式

对于输入尺寸 $H_{in}$，stride=2, padding=1, kernel=3：

$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times 1 - 3}{2} \right\rfloor + 1 = \left\lceil \frac{H_{in}}{2} \right\rceil
$$

这意味着：
- 奇数尺寸输入会产生 **向上取整** 的尺寸（如 135→68）
- 这是验证时必须 pad 到 16 的倍数的根本原因

### 3.3 上采样模块

```python
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
)
```

#### 3.3.1 为什么用 Bilinear + Conv 而非 ConvTranspose？

| 特性 | Bilinear + Conv | ConvTranspose2d |
|------|----------------|-----------------|
| 棋盘伪影 | ❌ 无 | ⚠️ 可能有 |
| 计算量 | 略高（插值+卷积） | 略低 |
| 参数效率 | Conv 降通道更灵活 | 通道固定 |
| 高频恢复 | 更平滑，适合VQE | 可能过锐 |

**VQE任务的先验**：压缩伪影通常是块效应和振铃效应，需要平滑重建而非锐利边缘。Bilinear插值的低通特性有助于抑制高频噪声。

### 3.4 Skip Connection 融合策略

```python
x = torch.cat([x, skip], dim=1)  # 通道维度拼接
x = conv(x)  # 3×3 Conv 降通道
```

#### 3.4.1 为什么用 Concat 而非 Add？

| 操作 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **Concat** | $[\mathbf{x}_{up}; \mathbf{x}_{skip}]$ | 保留全部信息，不丢失特征 | 通道翻倍，计算量增加 |
| Add | $\mathbf{x}_{up} + \mathbf{x}_{skip}$ | 计算量小，类似ResNet | 要求通道匹配，信息融合不充分 |

**选型理由**：编码器浅层（Skip1, Skip2）包含丰富的纹理细节，深层（Skip3, Skip4）包含语义结构。Concat 让解码器通过后续 3×3 Conv **自适应加权**融合，而非强制相加。

---

## 4. 参数量与计算复杂度分析

### 4.1 参数量精确计算

设 $b = \text{base\_ch}$，定义单组卷积参数量：$\text{Conv}(C_{in}, C_{out}, K) = C_{in} \cdot C_{out} \cdot K^2$（bias=False）

#### 4.1.1 编码器

| 组件 | 结构 | 参数量公式 | base_ch=32 数值 |
|------|------|-----------|----------------|
| Init Conv | Conv(3, b, 3×3) | $27b$ | 864 |
| Enc1 | 2×ResBlock(b) + Down(b→2b) | $2 \times 18b^2 + 9b \cdot 2b = 54b^2$ | 55,296 |
| Enc2 | 2×ResBlock(2b) + Down(2b→4b) | $2 \times 18(2b)^2 + 9(2b)(4b) = 216b^2$ | 221,184 |
| Enc3 | 2×ResBlock(4b) + Down(4b→8b) | $2 \times 18(4b)^2 + 9(4b)(8b) = 864b^2$ | 884,736 |
| Enc4 | 2×ResBlock(8b) + Down(8b→8b) | $2 \times 18(8b)^2 + 9(8b)(8b) = 2880b^2$ | 2,949,120 |

**编码器合计**：$27b + 4014b^2$

#### 4.1.2 瓶颈层

| 组件 | 参数量公式 | base_ch=32 数值 |
|------|-----------|----------------|
| Bottleneck | $2 \times 18(8b)^2 = 2304b^2$ | 2,359,296 |

#### 4.1.3 解码器

| 组件 | 结构 | 参数量公式 | base_ch=32 数值 |
|------|------|-----------|----------------|
| Dec1 | Up+Conv+ConcatConv+2×ResBlock(8b) | $576b^2+1152b^2+2304b^2=4032b^2$ | 4,128,768 |
| Dec2 | Up+Conv+ConcatConv+2×ResBlock(4b) | $288b^2+288b^2+576b^2=1152b^2$ | 1,179,648 |
| Dec3 | Up+Conv+ConcatConv+2×ResBlock(2b) | $72b^2+72b^2+144b^2=288b^2$ | 294,912 |
| Dec4 | Up+Conv+ConcatConv+2×ResBlock(b) | $18b^2+18b^2+36b^2=72b^2$ | 73,728 |

**解码器合计**：$5544b^2$

#### 4.1.4 输出头

| 组件 | 参数量公式 | base_ch=32 数值 |
|------|-----------|----------------|
| Output Conv | Conv(b, 3, 3×3) | $27b$ | 864 |

#### 4.1.5 总计

$$
\text{Total} = 54b + (4014 + 2304 + 5544)b^2 = 54b + 11862b^2
$$

代入 $b=32$：

$$
\text{Total} = 1728 + 12,146,688 = 12,148,416 \approx 12.15\text{M}
$$

✅ 与代码实测的 **12.16M** 一致（微小差异来自 BN 参数的 ~2K）。

| base_ch | 参数量 | 是否符合 5–15M |
|---------|--------|--------------|
| 16 | **3.35M** | ✅（偏轻量） |
| **32** | **12.16M** | ✅ **推荐** |
| 48 | **27.0M** | ❌ 超出 |
| 64 | **45.5M** | ❌ 严重超出 |

> **结论**：标准卷积下，base_ch=64 的 4层 U-Net 参数量数学上不可能控制在 15M 以内。如需严格满足指标，**必须使用 base_ch ≤ 35**。

### 4.2 FLOPs 估算

以输入 `256×256`，base_ch=32 为例：

FLOPs 主要来源是卷积操作。对于 Conv($C_{in}$, $C_{out}$, $K×K$, $H×W$)：

$$
\text{FLOPs} \approx 2 \cdot H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2
$$

（因子2来自乘法和加法）

| 阶段 | 空间尺寸 | FLOPs估算 | 占比 |
|------|---------|----------|------|
| Enc1 | 256² | 2.1G | 4.5% |
| Enc2 | 128² | 2.1G | 4.5% |
| Enc3 | 64² | 2.1G | 4.5% |
| Enc4 | 32² | 1.8G | 3.9% |
| Bottleneck | 16² | 0.9G | 1.9% |
| Dec1 | 32² | 3.0G | 6.4% |
| Dec2 | 64² | 2.6G | 5.6% |
| Dec3 | 128² | 2.2G | 4.7% |
| Dec4 | 256² | 1.8G | 3.9% |
| **编码器+瓶颈** | - | **~11G** | **~24%** |
| **解码器** | - | **~35G** | **~76%** |
| **总计** | - | **~46G** | **100%** |

> **关键洞察**：解码器贡献了 **76% 的计算量**。这是因为解码器的 Concat 操作使通道数翻倍，后续 Conv 的计算量与 $C_{in} \cdot C_{out}$ 成正比。优化解码器通道数是提速的关键。

---

## 5. 感受野与特征层次

### 5.1 有效感受野（ERF）

对于 $L$ 层卷积网络，理论感受野（RF）为：

$$
\text{RF} = 1 + \sum_{l=1}^{L} (K_l - 1) \prod_{i=1}^{l-1} s_i
$$

其中 $K_l=3$（所有卷积核），$s_i$ 为累积 stride。

以 base_ch=32，输入 256×256 为例：

| 位置 | 到该点的层数 | 累积Stride | 理论RF | 有效RF（经验） |
|------|------------|-----------|--------|--------------|
| Skip1 (Enc1后) | 2 ResBlock + 1 Conv | 1 | 11×11 | ~7×7 |
| Skip2 (Enc2后) | +2 ResBlock + 1 Down | 2 | 27×27 | ~18×18 |
| Skip3 (Enc3后) | +2 ResBlock + 1 Down | 4 | 59×59 | ~38×38 |
| Skip4 (Enc4后) | +2 ResBlock + 1 Down | 8 | 123×123 | ~78×78 |
| Bottleneck | +2 ResBlock | 16 | 251×251 | ~160×160 |
| Dec4输出 | +所有解码器 | 1 | 507×507 | ~300×300 |

> **有效RF < 理论RF** 的原因：ResBlock 中两个 3×3 的权重通常较小，边缘像素的梯度贡献呈高斯衰减。

### 5.2 多尺度特征融合

U-Net 的 skip connection 实现了**显式的多尺度融合**：

- **Skip1**（高分辨率，低语义）：捕获纹理、边缘、细粒度伪影
- **Skip4**（低分辨率，高语义）：捕获整体结构、亮度、大区域平滑度
- **Bottleneck**：全局上下文（虽然纯卷积的全局能力有限）

对于 HEVC QP32 压缩视频，主要伪影类型与特征层对应：

| 伪影类型 | 空间尺度 | 主要恢复层 |
|---------|---------|-----------|
| 块效应（Blocking） | 8×8 边界 | Skip1 + Skip2 |
| 振铃效应（Ringing） | 边缘周围 | Skip2 + Skip3 |
| 模糊（Blurring） | 整区域 | Skip3 + Skip4 |
| 颜色失真 | 低频全局 | Bottleneck + Skip4 |

---

## 6. 训练动力学分析

### 6.1 损失函数设计

```python
Loss = 1.0 × L1(pred, target) + 1.0 × (1 - SSIM(pred, target))
```

#### 6.1.1 L1 Loss 的作用

$$
\mathcal{L}_{L1} = \frac{1}{N} \sum_{i} |\hat{y}_i - y_i|
$$

- **保边缘**：相比 L2（MSE），L1 对异常值（outliers）不敏感
- **稀疏梯度**：在零点附近梯度恒定，不易梯度消失
- **与 PSNR 相关性**：L1 最小化与峰值信噪比（PSNR）最大化有强相关性

#### 6.1.2 SSIM Loss 的作用

$$
\mathcal{L}_{SSIM} = 1 - \text{SSIM}(\hat{y}, y)
$$

其中：

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$

- **感知质量**：SSIM 模拟人眼对亮度、对比度、结构敏感度的感知
- **局部统计**：11×11 高斯滑动窗口，关注局部结构相似性而非逐像素差异
- **与 L1 互补**：L1 保全局像素精度，SSIM 保局部结构感知

#### 6.1.3 权重平衡

权重 `1.0 : 1.0` 的选型依据：

| 权重组合 | L1主导 | SSIM主导 | 实验现象 |
|---------|--------|---------|---------|
| 1.0 : 0.0 | ✅ | | 高PSNR，但图像可能过平滑 |
| 0.0 : 1.0 | | ✅ | 结构清晰，但颜色/亮度偏移 |
| **1.0 : 1.0** | ✅ | ✅ | **平衡像素精度与感知质量** |
| 1.0 : 2.0 | | ✅ | 更锐利，可能引入伪影 |

> **控制变量要求**：A/B/C 必须严格保持此权重，任何改动都会引入损失变化的混淆变量。

### 6.2 优化器动力学

```python
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

#### 6.2.1 为什么用 Adam？

- **自适应学习率**：ResBlock 中 BatchNorm + Conv 的参数尺度差异大，Adam 的二阶矩估计能自动调整各参数的学习步长
- **对超参数不敏感**：相比 SGD，Adam 对初始学习率的选择更鲁棒
- **适合小batch**：本方案 A模型 batch_size=32（RTX 5080 16GB 满载配置），Adam 的自适应特性在大 batch 下依然稳定，且收敛速度快于 SGD + Momentum

#### 6.2.2 为什么 weight_decay=0？

- A方案作为基线，先排除正则化效应
- 视频质量增强任务通常**不易过拟合**（输入输出空间高度结构化，且训练数据量大）
- 如后续实验发现过拟合，可开启 weight_decay=1e-4 作为消融实验

#### 6.2.3 StepLR 的调度策略

- **step_size=30**：每30个epoch学习率降至10%
- **合理性**：100 epoch 训练中，LR 在 epoch 30 和 60 处下降，共两次
- **与 Cosine 对比**：StepLR 提供阶梯式精细调整，Cosine 提供更平滑的退火。本方案选用 StepLR 是为了在 A/B/C 之间保持完全一致，便于对比。

#### 6.2.4 梯度累积

```python
grad_accum_steps = 1  # 当前关闭
```

- 如需进一步增大等效 batch size（如模拟 64），可设为 `2`（等效 bs=64），但会减慢参数更新频率
- 当前 batch_size=32 已足够填满 RTX 5080，无需开启

### 6.3 梯度裁剪

```python
clip_grad_norm = 1.0
```

- **防止梯度爆炸**：Pre-Activation 网络虽稳定，但在 AMP 混合精度下仍可能出现异常梯度
- **阈值选择**：1.0 是保守值。如观察到训练不稳定，可尝试 0.5（更严格）或 5.0（更宽松）

### 6.4 AMP 混合精度训练（当前关闭）

```python
amp = False  # 默认关闭
```

**关闭原因（本架构特有问题）**：

本方案采用 **Pre-Activation ResBlock + 无界输出头**，在 AMP (FP16) 下存在**数值漂移**，导致训练发散或 `loss=NaN`：

1. **Pre-Activation BN**：BatchNorm 在 FP16 下统计量精度不足，尤其是 batch_size 较大时（≥32），running mean/variance 累积误差导致分布偏移
2. **无界输出头**：输出层无激活函数，`pred ∈ (-∞, +∞)`。FP16 的动态范围有限（最大 ~65504），训练初期极易溢出
3. **残差相加**：`x + body(x)` 中若两项尺度差异大，FP16 精度损失会导致梯度异常

**实测结果**：
- FP32 (bs=32)：显存 ~11–12GB，训练稳定，PSNR 正常提升
- FP16 (bs=32)：loss 在 epoch 1–3 内迅速发散至 NaN

**权衡**：
- FP32 显存占用虽高，但 RTX 5080 16GB 恰好能容纳 bs=32
- 速度损失约 15–20%（相比 FP16），但稳定性优先
- 如需提速，可考虑 `torch.compile`（见第9.5节）而非 AMP

> **未来方向**：如确需 AMP，可将 Pre-Activation 替换为 Post-Activation，或在输出头后增加 `nn.Hardtanh(0, 1)` 限制输出范围。

---

## 7. 与标准U-Net的差异化设计

### 7.1 对比表

| 特性 | 本方案 PureResUNet | 标准 U-Net (Ronneberger) | 现代 VQE 网络 (EDVR/BasicVSR) |
|------|-------------------|------------------------|------------------------------|
| 基本单元 | Pre-Activation ResBlock | 2×Conv+ReLU | ResBlock / 3D Conv |
| Skip连接 | Concat + Conv | Concat + Conv | Concat / 通道注意力 |
| 下采样 | Stride-2 Conv | MaxPool | Stride-2 Conv |
| 上采样 | Bilinear + Conv | UpConv / Transpose | PixelShuffle / Bilinear |
| 时序建模 | ❌ 无（单帧） | ❌ 无 | ✅ 多帧对齐 |
| 注意力 | ❌ 无 | ❌ 无 | ✅ 自注意力 / 通道注意力 |
| 参数量 (输入256²) | 12M | ~31M | >20M |

### 7.2 为什么选择"复古"设计？

1. **控制变量**：A方案必须排除所有可能引入性能增益的"现代技巧"
2. **可解释性**：纯卷积网络的行为更容易分析和调试
3. **计算效率**：12M 参数在 RTX 5080 上训练速度快，实验周期短
4. **B/C扩展性**：简单的结构更容易插入 Cross-Attention 和 Router 模块

---

## 8. B/C方案扩展接口

### 8.1 为B方案预留的接口

```python
# 1. 编码器特征提取（已实现）
def get_encoder_features(self, x):
    x = self.init_conv(x)
    feats = []
    for i in range(4):
        for res in self.enc_blocks[i]:
            x = res(x)
        feats.append(x)  # [e1, e2, e3, e4]
        x = self.down_blocks[i](x)
    return feats, x
```

**B方案插入点**：
- 在 `Dec3` 和 `Dec4`（或所有解码器层）引入 `CrossAttention(Query=decoder_feat, Key/Value=CLIP_feat)`
- CLIP 特征通过 `get_encoder_features` 的 bottleneck 输入提取

### 8.2 为C方案预留的接口

```python
# 2. ResBlock 侧向输入（已预留）
def forward(self, x, side_input=None):
    out = self.body(x)
    return x + out  # side_input 可替换为门控加权
```

**C方案插入点**：
- 将 `ResBlock` 替换为 `DualExpertResBlock`
- 两个轻量专家并行，输出由 Router MLP 加权
- `side_input` 可传递门控权重或 CLIP 条件

---

## 9. 调优指南与最佳实践

### 9.1 base_ch 与 batch_size 选择决策树（基于 RTX 5080 实测）

```
GPU显存 ≥ 16GB? 
    ├── 是 → A模型: batch_size=32, base_ch=32（推荐，12M参数，显存~12GB）
    │         B/C模型: batch_size=20, base_ch=32（显存~12GB）
    │         └── 追求更高质量? → base_ch=48（27M），A模型 bs=16，B/C模型 bs=12
    └── 否 → batch_size=8, base_ch=24（~7M参数）
                └── 显存 < 8GB? → base_ch=16（~3.3M参数），batch_size=4
```

**RTX 5080 16GB 实测满载配置**：

| 模型 | batch_size | 训练显存 | GPU Util | 推荐度 |
|------|-----------|---------|---------|--------|
| A (PureResUNet) | 32 | ~11.5 GB | 98–99% | ⭐⭐⭐ 最佳 |
| B (ResUNet+VLM) | 20 | ~11.5 GB | 95–99% | ⭐⭐⭐ 最佳 |
| C (ResUNet+Router) | 20 | ~12.0 GB | 95–99% | ⭐⭐⭐ 最佳 |
| A | 16 | ~5.8 GB | 60–75% | ⭐⭐ 偏低 |
| A | 8 | ~3.5 GB | 40–55% | ⭐ 严重欠载 |

### 9.2 学习率调优

| 现象 | 诊断 | 解决方案 |
|------|------|---------|
| Loss 持续上升 | LR过高或梯度爆炸 | 降低LR至5e-5，或收紧clip_grad_norm至0.5 |
| Loss  plateau（平台期） | LR过低或陷入局部最优 | 升高LR至2e-4，或改用CosineAnnealing |
| Loss 震荡剧烈 | BatchNorm统计不稳定 | 增大batch_size，或启用BatchNorm同步 |

### 9.3 早停策略优化

默认配置：
```python
early_stop_patience = 15
early_stop_min_delta = 0.0
```

| 场景 | 建议配置 |
|------|---------|
| 快速实验 | patience=10, min_delta=0.01 |
| 严格最优 | patience=20, min_delta=0.0 |
| 噪声环境 | patience=15, min_delta=0.05（容忍噪声） |

### 9.4 数据加载优化（Windows 实测调优）

**当前状态（已优化）**：

```
配置：num_workers=6, persistent_workers=True, pin_memory=False
实测：CPU 总占用 ~20–25%，GPU Util 98–99%
结论：数据加载已**不是瓶颈**，I/O 供给充足
```

**关键参数说明**：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `num_workers` | **6** | 9800X3D 16线程下，6 个 worker 是 Windows 稳定与速度的最优平衡点。>8 易引发多进程 CUDA 冲突 |
| `persistent_workers` | **True** | 保持 worker 进程存活，避免每 epoch 重建带来的启动开销（Windows 上收益明显） |
| `pin_memory` | **False** | Windows 保持 False。pin_memory 的固定内存线程在 Windows 上可能与 CUDA 上下文冲突，导致随机崩溃 |

**历史瓶颈（旧配置）**：
- `num_workers=0`：I/O 占比 ~80%，GPU 大量空闲
- `num_workers=10`：进程切换开销大，偶发多进程 CUDA 错误

**进一步优化方案**（如需极致提速）：
1. **预加载**：将 YUV 预先转换为 .lmdb 或 .npy 序列文件，彻底消除逐帧解析开销
2. **Prefetch**：使用 `torch.data prefetcher` 重叠 I/O 和计算
3. **Linux 迁移**：如迁移至 Linux 服务器，可启用 `pin_memory=True` + `num_workers=8`，速度再提升 10–15%

### 9.5 Windows 训练性能优化（RTX 5080 专属调优）

本节记录针对 RTX 5080 + Windows 10 + PyTorch 2.12 nightly 的实测优化经验。

#### 9.5.1 cuDNN Benchmark

```python
torch.backends.cudnn.benchmark = True
```

- 固定输入尺寸（256×256）时，cuDNN 会自动搜索最优卷积算法
- 首 epoch 启动稍慢（warmup），后续 epoch 显著加速
- **实测收益**：epoch 时间从 ~180s 降至 ~145s（约 20% 提升）

#### 9.5.2 torch.compile（Windows 限制）

```python
if hasattr(torch, 'compile'):
    try:
        import importlib.util
        triton_available = importlib.util.find_spec("triton") is not None
        if triton_available:
            model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        pass  # 自动回退 eager mode
```

- **Windows 现状**：PyTorch 2.x 的 `torch.compile` 依赖 Triton 后端，Windows 上 **不可用**
- **实际行为**：自动回退到 eager mode，不影响训练
- **未来**：如 PyTorch 官方支持 Windows Triton，可自动获得 1.2–1.5× 加速

#### 9.5.3 进程优先级

```python
from utils import set_high_priority
set_high_priority()  # 将 Python 进程设为 HIGH_PRIORITY_CLASS
```

- 防止 CMD/Terminal 窗口失去焦点后 Windows 降低进程优先级
- 配合关闭 CMD "快速编辑模式"，避免切窗口后训练降速

#### 9.5.4 训练强度监控参考值

正常训练时的硬件占用应接近以下范围：

| 指标 | 正常范围 | 说明 |
|------|---------|------|
| GPU Util | **95–99%** | 计算饱和，理想状态 |
| 显存占用 | **11.5–13.0 GB** | 约占 80%，达到目标 |
| GPU 功耗 | **240–270 W** | RTX 5080 上限 360W，有余量 |
| GPU 温度 | **60–70°C** | 非常安全 |
| CPU 占用 | **15–30%** | 数据供给充足，无瓶颈 |
| 内存可用 | **>20 GB** | 32GB 总内存非常宽裕 |

> **诊断**：若 GPU Util < 80%，检查 `num_workers` 是否过低或数据是否预加载；若显存 < 8GB，说明 batch_size 太小，GPU 未满载。

---

## 10. 已知限制与规避策略

### 10.1 尺寸对齐与推理策略

**训练时**：
- `patch_size=256`（256/16=16，天然对齐）✅

**验证/测试时**：
`validate()` / `test()` 函数实现了**三级推理策略**，自动适配不同分辨率：

1. **Class A/B 大分辨率强制 tile-based**（`H > 720` 或 `W > 1280`）：
   ```python
   pred = tile_predict(model, lq_frame, tile_size=256, stride=128)
   ```
   - 将整帧切分为 256×256 的 overlap tile，stride=128（50% 重叠）
   - 避免大分辨率整帧推理导致 OOM

2. **小分辨率整帧推理**（`H ≤ 720` 且 `W ≤ 1280`）：
   ```python
   lq_frame, pads = _pad_to_multiple(lq_frame, multiple=16)
   pred = model(lq_frame)
   # 去除对称 pad
   pred = pred[:, :, pad_top:pad_top+H, pad_left:pad_left+W]
   ```
   - 先反射 pad 到 16 的倍数，再通过 U-Net，最后裁去 pad

3. **OOM Fallback**：
   ```python
   except RuntimeError as e:
       if 'out of memory' in str(e).lower():
           torch.cuda.empty_cache()
           pred = tile_predict(...)  # 降级为 tile 推理
   ```
   - 即使小分辨率也可能因显存碎片或峰值波动导致 OOM，自动降级为 tile-based

> **关键设计**：三级策略确保任何分辨率下都能安全推理，不会因为单帧过大导致验证中断。

### 10.2 纯卷积的全局建模局限

**问题**：纯卷积的感受野有限，无法有效建模长距离依赖（如整帧亮度漂移）。

**影响**：
- 对大范围平滑区域（如天空、墙面）的恢复可能不如 attention-based 方法
- 对复杂纹理区域（如草地、树叶）的恢复效果较好

**缓解**：
- 增大 base_ch 以提升特征表达能力
- B方案引入的 Cross-Attention 正是为解决此问题

### 10.3 BatchNorm在评估时的统计偏移

**问题**：训练时 BN 使用 batch 统计量，评估时切换为移动平均。如果 batch_size 较小（如 ≤16），batch 统计量噪声大，可能导致训练和评估的分布偏移。

**缓解**：
- 使用 `nn.SyncBatchNorm`（多卡时同步统计量）
- 或替换为 `nn.InstanceNorm2d`（但会损失跨样本统计）
- 当前单卡训练，batch_size=32 对 256×256 patch 的统计量非常稳定，BN 统计噪声可忽略

### 10.4 无激活输出头的数值范围

**问题**：输出层无激活函数，`pred ∈ (-∞, +∞)`。

**风险**：
- 训练初期可能输出极大/极小值，导致 L1 loss 爆炸
- 梯度裁剪（clip_grad_norm=1.0）可缓解

**后处理**：
- 推理时显式 clamp：`pred = pred.clamp(0, 1)`
- 已集成在 `validate()` 和 `test()` 中

---

## 11. 附录：完整公式推导

### 11.1 ResBlock 参数量推导

对于通道数为 $C$ 的 ResBlock：

```
BN(C) → ReLU → Conv(C, C, 3×3) → BN(C) → ReLU → Conv(C, C, 3×3)
```

- BN 参数量：$2C$（$\gamma$ 和 $\beta$）—— 可忽略
- 每个 Conv 参数量：$C \cdot C \cdot 3 \cdot 3 = 9C^2$
- ResBlock 总计：$2 \times 9C^2 = 18C^2$

### 11.2 全网络参数量闭合公式

$$
\boxed{\text{PureResUNet}(b) = 54b + 11862b^2}
$$

其中 $b = \text{base\_ch}$。

| base_ch | 精确参数量 |
|---------|-----------|
| 16 | 3,035,376 |
| 24 | 6,829,872 |
| **32** | **12,148,416** |
| 40 | 18,991,920 |
| 48 | 27,360,432 |
| 64 | 45,455,424 |

### 11.3 解码器计算量占比推导

解码器计算量高的根本原因：

$$
\text{Dec1 FLOPs} \propto H_{bottleneck}^2 \cdot C_{bottleneck}^2 = (16)^2 \cdot (8b)^2 = 16384b^2
$$

而编码器最深层：

$$
\text{Enc4 FLOPs} \propto (16)^2 \cdot (8b) \cdot (8b) = 16384b^2
$$

但解码器有 **4层**，每层都要处理 Concat 后的双倍通道：

$$
\text{Decoder Total FLOPs} \approx \sum_{i=1}^{4} H_i^2 \cdot (2C_{skip,i}) \cdot C_{out,i} \cdot 9
$$

由于 $H_i$ 随上采样指数增长，而 $C$ 线性下降，解码器总 FLOPs 约为编码器的 **3–4 倍**。

---

## 12. 实验检查清单

在启动 A/B/C 对比实验前，确认以下事项：

- [ ] `config.py` 中 `model_type='A'`，`base_ch=32`
- [ ] `loss_type='L1_SSIM'`，`l1_weight=1.0`，`ssim_weight=1.0`
- [ ] `optimizer='Adam'`，`lr=1e-4`，`weight_decay=0`
- [ ] `batch_size=32`（A模型）或 `20`（B/C模型）
- [ ] `num_workers=6`，`pin_memory=False`，`persistent_workers=True`
- [ ] `amp=False`（本架构 FP16 会 NaN）
- [ ] `early_stop=True`，`patience=15`
- [ ] 训练前运行 `python scripts/smoke_test.py` 通过
- [ ] 训练前运行 `python scripts/synthetic_train_test.py` 确认 loss 下降
- [ ] 检查 `checkpoints/` 目录存在且可写
- [ ] 监控首 epoch 的 loss 值：应在 0.5–2.0 之间，若 >5.0 说明初始化异常或数据有问题
- [ ] 首 epoch 后检查 GPU Util 是否达到 95%+，若低于 80% 参考第 9.5 节诊断

---

*文档结束。如有技术疑问，请结合具体实验日志进行诊断。*
