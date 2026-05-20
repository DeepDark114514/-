# B 方案：DegFiLM-ResUNet 技术说明文档

> 作者：于陈远（南京信息工程大学 22 级信安 1班）
> 日期：2026.5.18
> 版本：v2.0（根据导师反馈修正 DegEstimator 下采样层数与 FiLM 数学形式）

---

## 一、设计动机：A 方案暴露了什么？

### 1.1 A 方案 Cross-QP 实验结果

| QP | Model PSNR | Base PSNR | **Gain (dB)** | Model SSIM | Base SSIM | Gain |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 22 | 36.79 | 36.62 | +0.17 | 0.9384 | 0.9346 | +0.0038 |
| 27 | 34.47 | 34.22 | +0.25 | 0.9113 | 0.9055 | +0.0059 |
| 32 | 32.39 | 32.04 | +0.35 | 0.8819 | 0.8715 | +0.0104 |
| 37 | 29.74 | 29.38 | +0.36 | 0.8266 | 0.8123 | +0.0144 |
| 42 | 27.87 | 27.49 | +0.38 | 0.7825 | 0.7625 | +0.0201 |

### 1.2 核心诊断

**关键观察**：QP32 → QP42，baseline PSNR 暴跌 **4.55 dB**（32.04 → 27.49），但模型增益只增加了 **0.03 dB**（0.35 → 0.38）。

这说明：**A 方案的修复策略近似固定，对输入质量"浑然不觉"**。

- 高 QP（重压缩）：伪影严重，可修复空间大 → 但 A 方案没有"加大修复力度"
- 低 QP（轻压缩）：伪影轻微，过度修复会损失细节 → 但 A 方案没有"减小修复力度"

**根本原因**：PureResUNet 是一个"盲修复"网络——它对不同 QP 的输入使用同一套卷积权重，修复力度无法根据损伤程度动态调节。

---

## 二、B 方案核心思想

> **给网络加一只"眼睛"（退化估计器），装一组"旋钮"（FiLM 条件层），让网络"看"到输入质量后，自适应调节修复力度。**

**两个参考文献的启发**：

1. **PromptCIR (CVPRW 2024, NTIRE 2024 第一名)**：
   - 核心论点："预测数值 QF 缺乏空间信息，阻碍网络对图像内容的适应性"
   - 解决方案：**不预测显式 QF**，用轻量级 Prompt **隐式编码**压缩信息
   - Prompt 与图像特征动态交互，提供内容感知 + 失真感知的指导

2. **FiLM (AAAI 2018)**：
   - 全称：Feature-wise Linear Modulation
   - 核心：对神经网络中间特征做**逐通道仿射变换** `γ·F + β`
   - γ 和 β 从任意条件输入生成，是一种通用条件化层

**B 方案的本地化适配**：
- 用 **DegEstimator** 实现 PromptCIR 的"隐式退化编码"思想
- 用 **FiLM** 实现条件化残差调节
- 两者结合，端到端联合训练，无需预训练

---

## 三、核心改进详解（3 点）

### 3.1 改进 1：轻量退化估计器（DegEstimator）

**作用**：给网络加一只"眼睛"，让它能"看"到输入帧的退化程度。

**位置**：输入帧旁路，只运行一次，供深层复用。

**结构**：
```
Input (B, 3, H, W)
    ↓ Conv 3×3, stride=2, 3→16 channels
Feature (B, 16, H/2, W/2)
    ↓ Conv 3×3, stride=2, 16→32 channels
Feature (B, 32, H/4, W/4)
    ↓ Conv 3×3, stride=2, 32→48 channels
Feature (B, 48, H/8, W/8)
    ↓ Global Average Pooling
Feature (B, 48, 1, 1)
    ↓ Flatten → Linear(48 → 64)
DegEmbed d (B, 64)
```

**关键设计选择**：

| 设计 | 取值 | 理由 |
|------|------|------|
| 下采样层数 | **3 层**（1/8）| 导师要求。4 层（1/16）会丢失过多空间信息，尤其是块效应的周期结构 |
| 通道数 | 16→32→48 | 轻量级，总参数量仅 ~22K |
| 输出维度 | 64 | 足够表达退化信息，又不增加 FiLM 生成器的负担 |
| 是否预训练 | **否** | 端到端联合训练，让 DegEstimator 自动学习对修复有用的退化表征 |
| 是否冻结 | **否** | 冻结会切断梯度，退化表征无法随主任务优化 |
| 是否输出 QP 数字 | **否** | 遵循 PromptCIR：隐式编码优于显式预测 |

**参数量**：~22K（0.02M），仅占总参数的 **0.18%**。

---

### 3.2 改进 2：FiLM 条件层注入残差块

**作用**：给网络装一组"旋钮"，根据退化程度调节每个残差块的修复力度。

**数学形式**：
```
Output = x + (1 + γ) · body_out + β
```

**参数语义**：

| 参数 | 作用 | 物理意义 |
|------|------|---------|
| γ | 控制残差出力大小 | γ > 0：增强修复（重压缩时多修）；γ < 0：减弱修复（轻压缩时少修）；γ = 0：标准残差块 |
| β | 控制特征分布平移 | 防止缩放后分布漂移，避开 ReLU 死亡区 |

**为什么用 `(1 + γ)` 而不是 `γ`**：

这是本方案**最关键的数学设计**。

| 形式 | γ = 0 时的行为 | 训练初期效果 | 问题 |
|------|---------------|-------------|------|
| `x + γ·body_out + β` | `x + 0 = x`（恒等映射）| 残差通路被**完全关闭** | 训练初期主干网络学不到任何东西 |
| `x + (1+γ)·body_out + β` | `x + body_out`（标准 ResBlock）| **不干扰主干**，正常训练 | ✅ 正确 |

**生成方式**：退化嵌入 `d (B, 64)` 过两个轻量 Linear 层分别生成 γ, β：
```python
film_params = Linear(64 → 2*C)(d)   # 前 C 维 = γ, 后 C 维 = β
gamma = film_params[:, :C].view(B, C, 1, 1)
beta  = film_params[:, C:].view(B, C, 1, 1)
```

**初始化**：
```python
nn.init.zeros_(fc.weight)
nn.init.zeros_(fc.bias)
```
初始化后 γ=0, β=0，训练初期 FiLMResBlock 近似标准 ResBlock，**不干扰主干网络的学习**。

---

### 3.3 改进 3：分层插入策略（Encoder 不插，深层才插）

**插入位置**：

| 层级 | 块数 | 是否插入 FiLM | 理由 |
|------|------|-------------|------|
| Encoder level 0 (1×) | 2 ResBlock | ❌ 不插 | 最浅层负责边缘/颜色，退化感知不必要 |
| Encoder level 1 (2×) | 2 ResBlock | ❌ 不插 | 浅层特征应保持稳定，不受退化信息干扰 |
| Encoder level 2 (4×) | 2 ResBlock | ❌ 不插 | 中层特征开始抽象，但仍以内容为主 |
| Encoder level 3 (8×) | 2 ResBlock | ❌ 不插 | 深层编码器，但遵循 PromptCIR 的"decoder 导向"设计 |
| **Bottleneck (8×)** | **2 ResBlock** | ✅ **插入** | 感受野最大，需退化指导全局语义 |
| **Decoder level 0 (8×)** | **2 ResBlock** | ✅ **插入** | 重建阶段起始，需退化指导恢复力度 |
| **Decoder level 1 (4×)** | **2 ResBlock** | ✅ **插入** | 逐步上采样，继续退化感知重建 |
| **Decoder level 2 (2×)** | **2 ResBlock** | ✅ **插入** | 接近输出，精细调节修复强度 |
| Decoder level 3 (1×) | 2 ResBlock | ❌ 不插 | 最浅层最高分辨率，只负责细节精修 |

**总计**：8 个 FiLMResBlock（Bottleneck 2 + Decoder 前 3 层各 2）。

**为什么 Encoder 不插**：
1. 编码器负责提取内容特征（边缘、纹理、语义），不应被退化信息"污染"
2. PromptCIR 也是在 Decoder 阶段使用 Prompt，Encoder 不直接受 Prompt 影响
3. 减少参数开销，避免退化信息在浅层过早干预

---

## 四、模型架构总览

```
Input (B, 3, H, W)
    │
    ├──→ [DegEstimator] ──→ DegEmbed d (B, 64) ─────────────┐
    │                                                        │
    ↓                                                        │
Init Conv (3→32)                                           │
    │                                                        │
Encoder Level 0 (32ch) ──→ Skip[0] ────────────────────────┤
    ↓ Down                                                   │
Encoder Level 1 (64ch) ──→ Skip[1] ────────────────────────┤
    ↓ Down                                                   │
Encoder Level 2 (128ch) ──→ Skip[2] ───────────────────────┤
    ↓ Down                                                   │
Encoder Level 3 (256ch) ──→ Skip[3] ───────────────────────┤
    ↓ Down                                                   │
Bottleneck (256ch)                                         │
    ├─ FiLMResBlock + FiLM(d) ←────────────────────────────┤
    └─ FiLMResBlock + FiLM(d) ←────────────────────────────┘
    ↑ Up
Decoder Level 0 (256ch) + Skip[3]
    ├─ Conv (cat+conv)                                       │
    ├─ FiLMResBlock + FiLM(d) ←────────────────────────────┤
    └─ FiLMResBlock + FiLM(d) ←────────────────────────────┘
    ↑ Up
Decoder Level 1 (128ch) + Skip[2]
    ├─ Conv (cat+conv)                                       │
    ├─ FiLMResBlock + FiLM(d) ←────────────────────────────┤
    └─ FiLMResBlock + FiLM(d) ←────────────────────────────┘
    ↑ Up
Decoder Level 2 (64ch) + Skip[1]
    ├─ Conv (cat+conv)                                       │
    ├─ FiLMResBlock + FiLM(d) ←────────────────────────────┤
    └─ FiLMResBlock + FiLM(d) ←────────────────────────────┘
    ↑ Up
Decoder Level 3 (32ch) + Skip[0]
    ├─ Conv (cat+conv)
    ├─ ResBlock (标准)
    └─ ResBlock (标准)
    │
Out Conv (32→3)
    │
Output = OutConv + Input (全局残差学习)
```

---

## 五、与文献的精确对应关系

### 5.1 vs PromptCIR (CVPRW 2024)

| PromptCIR 原文 | B 方案的本地化实现 |
|---------------|------------------|
| "lightweight prompts to implicitly encode compression information" | DegEstimator：轻量 CNN 旁路，隐式编码退化信息 |
| "prompts directly interact with soft weights generated from image features" | FiLM 生成的 γ, β 与残差块输出的特征做逐通道仿射变换 |
| "dynamic content-aware and distortion-aware guidance" | 同一 batch 中不同样本（不同 QP）生成不同的 γ, β，实现样本级自适应 |
| "minimal parameter overhead" | DegEstimator 仅 22K，FiLM 总增量 205K，合计 **+1.69%** |
| Decoder 阶段注入 Prompt，Encoder 不变 | 完全一致：Encoder 不插，Bottleneck + Decoder 前 3 层插入 |

### 5.2 vs FiLM (AAAI 2018)

| FiLM 原文 | B 方案的本地化实现 |
|----------|------------------|
| "feature-wise affine transformation: γ·F + β" | 残差路径上的 `(1+γ)·body_out + β` |
| "conditioned on an arbitrary input" | 以 DegEstimator 输出的退化嵌入 `d` 作为条件输入 |
| "general-purpose conditioning method" | 不修改网络 backbone，仅在 ResBlock 中注入条件层 |
| "robust to ablations" | 初始化 γ=0, β=0，训练初期不干扰主干 |

---

## 六、与 A 方案的对比

| 维度 | A 方案 PureResUNet | B 方案 DegFiLMResUNet |
|------|-------------------|----------------------|
| **参数量** | 12.16M | 12.36M (+1.69%) |
| **条件化能力** | ❌ 无（盲修复） | ✅ 有（退化感知） |
| **修复策略** | 固定卷积权重，一刀切 | 根据输入质量动态调节 |
| **Encoder** | 4 层标准 ResBlock | 4 层标准 ResBlock（不变） |
| **Bottleneck** | 2 层标准 ResBlock | 2 层 **FiLMResBlock** |
| **Decoder** | 4 层标准 ResBlock | level 0-2: FiLMResBlock, level 3: 标准 ResBlock |
| **训练方式** | 端到端 | 端到端（DegEstimator 与主干一起训练） |
| **预训练** | 无需 | 无需 |

---

## 七、实现细节

### 7.1 文件结构

```
models/
├── base_unet.py          # BaseUNet 骨架（A/B 共用）
├── pure_resunet.py       # A 方案：PureResUNet
├── deg_film_blocks.py    # B 方案核心：DegEstimator + FiLM + FiLMResBlock
└── degfilm_resunet.py    # B 方案：DegFiLMResUNet
```

### 7.2 关键代码片段

**DegEstimator**（`deg_film_blocks.py`）：
```python
class DegEstimator(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(48, embed_dim)
```

**FiLMResBlock**（`deg_film_blocks.py`）：
```python
def forward(self, x, film_params=None):
    out = self.body(x)
    if film_params is not None:
        B, C = out.size(0), out.size(1)
        gamma = film_params[:, :C].view(B, C, 1, 1)
        beta  = film_params[:, C:].view(B, C, 1, 1)
        out = (1 + gamma) * out + beta   # 关键：1+γ 而非 γ
    return x + out
```

**DegFiLMResUNet 前向传播**（`degfilm_resunet.py`）：
```python
def forward(self, x):
    identity = x
    x = self.init_conv(x)
    
    # Encoder（不插 FiLM）
    skips = []
    for i in range(4):
        for res in self.enc_blocks[i]:
            x = res(x)
        skips.append(x)
        x = self.down_blocks[i](x)
    
    # 退化嵌入（只算一次）
    deg_embed = self.deg_estimator(identity)
    
    # Bottleneck（插入 FiLM）
    for idx, res in enumerate(self.bottleneck):
        film_params = self.film_bottleneck[idx](deg_embed)
        x = res(x, film_params)
    
    # Decoder（前 3 层插入 FiLM）
    for i in range(4):
        x = self.up_blocks[i](x)
        skip = skips[3 - i]
        if i < 3:
            x = self.dec_blocks[i][0](torch.cat([x, skip], dim=1))
            for j in range(2):
                film_params = self.film_decoder[i][j](deg_embed)
                x = self.dec_blocks[i][j + 1](x, film_params)
        else:
            x = self._fuse_skip(x, skip, i)  # 最浅层标准处理
    
    x = self.out_conv(x)
    return x + identity
```

### 7.3 训练配置（与 A 方案完全一致，单一变量原则）

```python
{
    'model_type': 'B',
    'base_ch': 32,
    'loss_type': 'L1_SSIM',
    'l1_weight': 1.0,
    'ssim_weight': 1.0,
    'optimizer': 'Adam',
    'lr': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'qp_list': [22, 32, 42],   # 训练时混合多 QP
    'val_qp': 32,              # 验证基准 QP
}
```

**启动命令**：
```bash
python train.py -m B --base_ch 32 --epochs 100
```

---

## 八、预期效果与验证计划

### 8.1 理论上能解决的问题

A 方案的核心问题是"修复策略固定"。B 方案的预期改善：

```
高 QP（重压缩，如 QP42）
    → DegEstimator 提取到"强退化"嵌入 d
    → FiLM 生成较大的 γ
    → 残差出力增强：(1+γ)·body_out
    → 修复更积极 → 增益提升（如 +0.45 dB）

低 QP（轻压缩，如 QP22）
    → DegEstimator 提取到"弱退化"嵌入 d
    → FiLM 生成较小的 γ（甚至负值）
    → 残差出力减弱：(1+γ)·body_out ≈ body_out
    → 修复更保守 → 避免过修复导致的细节损失
```

### 8.2 关键验证指标

| 指标 | A 方案表现 | B 方案预期 |
|------|-----------|-----------|
| Cross-QP 增益单调性 | 增益几乎不随 QP 变化 | 增益应随 QP 明显递增 |
| QP22 增益 | +0.17 dB | 保持稳定或略降（避免过修复） |
| QP42 增益 | +0.38 dB | **明显提升**（模型学会重压缩时多修） |
| 增益波动原因 | "模型对输入质量无感知" | "模型已感知，波动反映物理可修复空间" |

### 8.3 消融实验计划（训练完成后）

1. **γ 可视化**：统计不同 QP 下各 FiLM 层的 γ 分布，验证"高 QP → γ 更大"的假设
2. **DegEstimator 旁路**：固定 DegEstimator 输出为 0，观察模型是否退化为 A 方案性能
3. **Encoder 插入 FiLM**：对比 Encoder 也插 FiLM 的效果，验证"Encoder 不插"的合理性

---

## 九、与导师方案的差异化

| 导师方案（文献中的做法） | 本方案（轻量化本地适配） |
|------------------------|------------------------|
| 显式 DRL 预训练 + 冻结 | ❌ 不预训练、不冻结，端到端联合训练 |
| 多级 Stage + 分层终止 | ❌ 单级静态网络，简单直接 |
| STDA 空间-通道联合调制 | ❌ 只调通道统计量（γ, β），不搞空间注意力 |
| 预测 QP 数字作为条件 | ❌ 隐式编码退化信息，不输出数值 QF |
| 复杂控制器网络 | ❌ DegEstimator 仅 22K，FiLM 仅 Linear 层 |

**核心差异**：走轻量路线，在 A 方案基础上**最小侵入式**地添加条件化能力，总参数量增加 **< 2%**。

---

## 十、总结

B 方案 DegFiLM-ResUNet 通过两个轻量模块（DegEstimator + FiLM）解决了 A 方案"修复策略固定"的核心问题：

1. **DegEstimator** 让网络"看"到输入质量（PromptCIR 的隐式退化编码思想）
2. **FiLM** 让网络根据退化程度"调节"修复力度（AAAI 2018 的通用条件化层）
3. **分层插入策略** 确保退化信息只在需要的深层起作用（Encoder 不插，Bottleneck + Decoder 前 3 层插）

**关键数学设计**：`Output = x + (1 + γ) · body_out + β`，初始化 γ=0 时退化为标准 ResBlock，训练初期不干扰主干。

**预期效果**：Cross-QP 增益曲线更加合理——高 QP 时增益明显提升，低 QP 时避免过修复，模型真正做到了"对症下药"。
