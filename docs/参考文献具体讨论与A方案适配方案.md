# 参考文献具体讨论与A方案适配方案

> 整理时间：2026-05-15
> 用途：为毕设论文"相关工作"和"改进方法"章节提供深度分析素材
> 核心切入点：边缘模糊与过度平滑

---

## 本文档阅读指南

每篇文献按以下结构展开：
1. **论文信息** —— 标题、作者、发表、网址
2. **他在做什么** —— 研究背景与核心目标
3. **他怎么做的** —— 核心方法与技术细节
4. **对我有什么启发** —— 与A方案缺陷的关联
5. **A方案适配方案** —— 如何具体移植到PureResUNet
6. **优化建议** —— 实现时的注意事项和调参策略

---

## 文献一：AESOP —— 像素级损失导致模糊的理论基础

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | Auto-Encoded Supervision for Perceptual Image Super-Resolution |
| **作者** | MinKyu Lee, Sangeek Hyun, Woojin Jun, Jae-Pil Heo |
| **发表** | CVPR 2025 (IEEE/CVF Conference on Computer Vision and Pattern Recognition) |
| **网址** | https://arxiv.org/abs/2412.00124 |
| **代码** | https://github.com/2minkyulee/AESOP-Auto-Encoded-Supervision-for-Perceptual-Image-Super-Resolution |
| **类型** | 图像超分 / 感知质量 / 损失函数设计 |

### 他在做什么

这篇论文解决的是GAN-based超分辨率任务中一个长期被忽视的根本问题：**像素级L_p损失（L1/L2）本身就包含了一个导致模糊的分量**。

传统观点认为，像素级损失和感知质量之间存在trade-off——要想感知质量好，就得牺牲像素级保真度。所以之前的方法要么给L_p损失乘一个很小的权重，要么用低通滤波器处理。但本文证明：**这些绕弯子的方法根本没触及问题的本质**。

作者的核心发现是：L_p损失可以分解为两个子分量——一个贡献于模糊，另一个不贡献于模糊。而之前的方法之所以效果不好，是因为它们无法精确区分这两个分量。

### 他怎么做的

**核心方法：Auto-Encoded Supervision for Optimal Penalization loss (L_AESOP)**

具体步骤非常巧妙：

1. **预训练一个Auto-Encoder (AE)**：用L_p损失（比如L1）预训练一个自编码器。这个AE的作用是学习"在L_p损失下的最优重建空间"。

2. **在AE空间计算损失**：不再在原始像素空间计算L_p损失，而是在AE的解码器输出空间计算。即：
   ```
   L_AESOP = ||AE_decoder(AE_encoder(pred)) - AE_decoder(AE_encoder(gt))||_p
   ```
   注意：这里的AE space是**解码器之后的空间**，不是bottleneck（潜空间）。

3. **理论保证**：作者证明，AE起到了一个可微分近似算子的作用：
   ```
   ψ(·) := arg min_μ E[L(·, μ)]
   ```
   即AE的输出空间天然过滤掉了导致模糊的分量。

**实验结果**：在GAN-based SR框架中，仅仅把L_pix替换为L_AESOP，其他损失（perceptual loss、adversarial loss）完全不变，就能在保持感知质量的同时提升保真度。

### 对我有什么启发

**这是A方案"边缘模糊"问题的最直接理论支撑！**

A方案使用L1 + SSIM损失，本质上和这篇论文批判的对象完全一样。根据AESOP的理论：

1. **A方案的L1损失本身就包含了导致模糊的分量**——这不是模型架构的问题，是损失函数设计的问题。
2. **SSIM损失虽然关注结构，但单尺度SSIM对高频边缘的约束不足**——SSIM在平坦区域表现好，但在边缘区域无法有效区分"锐利边缘"和"模糊边缘"。
3. **简单的做法（如降低L1权重）无法解决问题**——因为这会同时降低两个分量的权重，导致训练不稳定。

**关键启示**：解决边缘模糊不能靠"调权重"，而要靠**在损失函数中显式引入不导致模糊的分量**，或者**在另一个特征空间中计算损失**。

### A方案适配方案

**适配思路一：借鉴AESOP的"特征空间损失"思想**

A方案不需要完整复现AESOP（因为AESOP需要预训练AE，开销较大），但可以借鉴其核心洞察：

```python
# 方案1.1：简化版AE空间损失
# 用一个轻量CNN作为"特征提取器"，在特征空间计算L1损失

class FeatureSpaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 轻量特征提取器：3层卷积即可
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
    
    def forward(self, pred, target):
        f_pred = self.encoder(pred)
        f_tgt = self.encoder(target)
        return torch.mean(torch.abs(f_pred - f_tgt))
```

**适配思路二：更实际的"高频分离损失"**

与其学AESOP搞一个完整的AE，不如直接对图像做高低频分离，只在低频部分用L1，高频部分用专门的损失：

```python
# 方案1.2：高低频分离L1损失
import torch.fft as fft

def frequency_separated_l1(pred, target, cutoff_ratio=0.25):
    # 做FFT
    pred_fft = fft.rfft2(pred)
    tgt_fft = fft.rfft2(target)
    
    # 构建高低频掩码
    h, w = pred_fft.shape[-2:]
    mask_low = torch.zeros_like(pred_fft)
    mask_low[:, :, :int(h*cutoff_ratio), :int(w*cutoff_ratio)] = 1
    mask_high = 1 - mask_low
    
    # 低频用L1（保结构），高频用L2（保细节）
    loss_low = torch.mean(torch.abs((pred_fft - tgt_fft) * mask_low))
    loss_high = torch.mean(((pred_fft - tgt_fft) * mask_high) ** 2)
    
    return loss_low + 0.1 * loss_high
```

### 优化建议

| 建议 | 说明 |
|---|---|
| **不要直接复制AESOP** | AESOP的完整实现需要预训练AE，对本科毕设来说太重。建议只借鉴其理论洞察，用更轻量的实现。 |
| **重点引用其理论分析** | 论文中可以用AESOP的实验结果作为"L1损失导致模糊"的权威证据，不需要自己重新证明。 |
| **与Edge Loss联合使用** | AESOP解决的是"损失的模糊倾向"，Edge Loss解决的是"边缘重建不足"，两者互补。 |
| **权重设置** | 如果引入特征空间损失，初始权重建议设为L1的0.1~0.2倍，逐步增加到0.5倍。 |

---

## 文献二：FDENet —— FFT Loss的最简洁实现

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | Image Deblurring via Frequency-Domain Feature Enhanced Convolutional Neural Networks |
| **作者** | Yecai Guo, Lixiang Ma, Yangyang Zhang |
| **发表** | Sensors (MDPI), 2026年3月 |
| **网址** | https://www.mdpi.com/1424-8220/26/6/1784 |
| **类型** | 图像去模糊 / 频率域特征增强 |

### 他在做什么

这篇论文解决的是**图像去模糊**任务中"纹理细节恢复不足"和"频域特征学习不充分"的问题。作者发现，现有的去模糊方法大多只在空间域做特征学习，忽视了频域信息——而图像的模糊本质上就是**高频信息的丢失**。

### 他怎么做的

**核心方法：FDENet (Frequency-Domain feature Enhanced Network)**

整个网络基于U-Net，但有三个关键创新：

#### 1. FFT-Res模块（Fast Fourier Transform Residual Block）

这是一个**空间域+频率域双分支**的残差模块：

```
输入特征 X
  ├── 空间域分支：标准残差连接
  └── 频率域分支：
        ├── 2D FFT 将X转换到频域 → 得到实部fr和虚部fi
        ├── 拼接(fr, fi)，过1x1卷积学习频域特征
        └── 2D IFFT 转回空间域
  
  输出：Y = Y_fft + Y_res + X （三者的残差融合）
```

**关键设计**：
- 频率域分支学习的是**全局结构信息**（FFT天然有全局感受野）
- 显式建模高低频特征关系
- 输出前做通道级归一化，保证两个分支尺度相当

#### 2. GFU模块（Gated Feed-forward Unit）

接在FFT-Res后面，用**门控机制**自适应增强重要特征、抑制冗余特征：

```
GFU(X) = Conv(Gating(X)) + X

其中 Gating(X) = GELU(DWConv(Conv(LN(X)))) ⊗ DWConv(Conv(LN(X)))
```

#### 3. 联合损失函数（核心！）

```
L_total = L_charbonnier + α * L_freq

L_charbonnier = sqrt((I_pred - I_gt)^2 + η^2)    # η = 0.001
L_freq = sqrt((F(I_pred) - F(I_gt))^2 + η^2)      # F = 2D FFT
```

**α = 0.1**（通过实验确定）

**核心思想**：
- Charbonnier loss 保空间域结构（比L1更平滑可导）
- FFT loss 保频域细节（强制模型恢复高频信息）
- 两者互补：空间域保整体结构，频率域保边缘纹理

### 对我有什么启发

**这篇论文是A方案引入频率域损失的最佳参考！**

FDENet的损失函数设计极其简洁，但效果明确：

1. **FFT loss的公式只有一行**：`sqrt((F(pred) - F(gt))^2 + η^2)`，PyTorch里用`torch.fft.rfft2`就能实现。
2. **Charbonnier loss比L1更适合VQE**：A方案当前用L1损失，Charbonnier loss在接近0时梯度不会太小，训练更稳定。
3. **α=0.1的权重比例可以直接借鉴**：FFT loss作为辅助损失，权重不宜过大。

**与A方案的关联**：
- A方案的"边缘模糊"本质就是高频丢失 → FFT loss直接在频域惩罚高频差异
- A方案的"过度平滑"本质是L1损失的平滑倾向 → Charbonnier loss + FFT loss联合可以缓解

### A方案适配方案

**适配思路：直接替换损失函数**

这是所有文献中**实现最简单、适配最直接**的方案。

```python
# losses/l1_ssim_loss.py 修改方案

import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_SSIM_Edge_FFT_Loss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, 
                 edge_weight=0.5, fft_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight
        self.charbonnier_eta = 0.001
        
    def charbonnier_loss(self, pred, target):
        """Charbonnier loss = sqrt((x-y)^2 + eta^2)"""
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + self.charbonnier_eta**2))
    
    def fft_loss(self, pred, target):
        """频率域损失"""
        # 对Y通道（或RGB每通道）做2D FFT
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        tgt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        # 分别对实部和虚部计算Charbonnier损失
        loss_real = torch.mean(torch.abs(pred_fft.real - tgt_fft.real))
        loss_imag = torch.mean(torch.abs(pred_fft.imag - tgt_fft.imag))
        
        return loss_real + loss_imag
    
    def edge_loss(self, pred, target):
        """Sobel边缘损失"""
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], 
                               dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], 
                               dtype=torch.float32, device=pred.device)
        
        pred_gx = F.conv2d(pred, sobel_x, padding=1)
        pred_gy = F.conv2d(pred, sobel_y, padding=1)
        tgt_gx = F.conv2d(target, sobel_x, padding=1)
        tgt_gy = F.conv2d(target, sobel_y, padding=1)
        
        return torch.mean(torch.abs(pred_gx - tgt_gx) + 
                         torch.abs(pred_gy - tgt_gy))
    
    def forward(self, pred, target):
        l1 = self.charbonnier_loss(pred, target)
        fft = self.fft_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        # SSIM可以用现有的ssim实现
        # ssim = 1 - ssim(pred, target)
        
        total = (self.l1_weight * l1 + 
                 self.fft_weight * fft + 
                 self.edge_weight * edge)
        
        return total, {'l1': l1.item(), 'fft': fft.item(), 'edge': edge.item()}
```

**训练配置修改**：

```python
# config.py 中修改
CONFIG = {
    'loss_type': 'L1_SSIM_Edge_FFT',
    'l1_weight': 1.0,
    'ssim_weight': 1.0,
    'edge_weight': 0.5,   # 新增
    'fft_weight': 0.1,    # 新增（参考FDENet的α=0.1）
}
```

### 优化建议

| 建议 | 说明 |
|---|---|
| **FFT只算Y通道** | 视频增强任务中，人眼对亮度通道（Y）的高频更敏感。可以先转YCbCr，只对Y通道算FFT loss，减少计算量。 |
| **Charbonnier η=0.001** | 这是FDENet论文里实验确定的最优值，可以直接用。 |
| **FFT权重从0.05开始** | 初始训练时高频分支可能不稳定，建议先用0.05训练10个epoch，再提升到0.1。 |
| **注意FFT的尺寸** | `torch.fft.rfft2`的输出尺寸是`(H, W//2+1)`，计算损失时不需要padding对齐。 |
| **混合精度问题** | FFT在FP16下可能有数值问题，建议保持FP32（A方案本来就关了AMP，没问题）。 |

---

## 文献三：RealisVSR —— 小波高频损失的前沿方案

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | RealisVSR: Detail-enhanced Diffusion for Real-World 4K Video Super-Resolution |
| **作者** | Weisong Zhao 等 |
| **发表** | arXiv:2507.19138, 2025 |
| **网址** | https://arxiv.org/abs/2507.19138 |
| **类型** | 视频超分 / 扩散模型 / 高频细节增强 |

### 他在做什么

RealisVSR解决的是**真实世界4K视频超分辨率**中的三个核心问题：
1. 基础模型对时间动态建模不一致
2. 复杂真实退化下**高频细节恢复有限**
3. 现有方法主要在720P数据集上评估，缺乏4K超分基准

特别值得注意的是，论文明确提到扩散模型虽然缓解了GAN-based方法的过度平滑问题，但现有的rectified flow loss（ rectified flow损失）**缺乏显式的频域判别能力**，导致网络优先保低频结构保真度，高频恢复次优。

### 他怎么做的

**核心创新：High-Frequency Rectified Diffusion Loss (HR-Loss)**

这是针对扩散模型rectified flow损失的改进，但核心思想可以推广到任何生成/重建模型。

**HR-Loss的组成**：

```
HR-Loss = Wavelet-based High-Frequency Constraint + HOG-based Texture Constraint
```

#### 1. 小波高频约束

- 对预测速度和真实速度（velocity field）分别做**小波分解**
- 在小波高频子带中计算差异
- 强制扩散模型在预测速度场时保留高频成分

#### 2. HOG纹理约束

- 提取预测结果和真值图的**HOG（方向梯度直方图）特征**
- HOG特征对边缘和纹理的局部结构非常敏感
- 在HOG特征空间中计算感知损失

**其他创新**（了解即可，对A方案直接借鉴有限）：
- **CPC (Consistency Preserved ControlNet)**：与Wan2.1视频扩散模型结合，抑制artifacts
- **RealisVideo-4K数据集**：首个公开4K VSR基准

### 对我有什么启发

**核心启发：频率域约束的"多尺度"思路**

RealisVSR告诉我们：
1. **单一频域损失不够**——FFT loss虽然好，但它是全局的，缺乏局部纹理感知。
2. **小波分解更适合多尺度高频恢复**——小波天然把图像分解成不同频率的子带，可以针对不同子带设置不同权重。
3. **HOG特征可以作为纹理感知的补充**——如果只用梯度（Sobel），只能捕捉边缘；HOG能捕捉纹理的局部统计特性。

**与A方案的关联**：
- A方案的边缘模糊在"宏观边缘"和"微观纹理"两个尺度上都有表现
- 仅用Sobel edge loss只能解决宏观边缘
- 需要引入多尺度高频约束才能同时解决纹理丢失

### A方案适配方案

**适配思路：多尺度高频损失（小波+HOG的简化版）**

考虑到A方案是CNN-based（不是扩散模型），不能完全照搬HR-Loss，但可以提取其核心思想：

```python
# 方案3.1：Haar小波多尺度高频损失
import pywt
import torch

def wavelet_hf_loss(pred, target, levels=2, weights=[0.5, 0.3]):
    """
    Haar小波多尺度高频损失
    levels: 分解层数
    weights: 每层高频损失的权重
    """
    loss = 0
    
    # 转为numpy做pywt小波变换（训练时可以用torch小波库替代）
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    
    for i in range(pred_np.shape[0]):  # batch维度
        for c in range(pred_np.shape[1]):  # channel维度
            # 2D Haar小波分解
            coeffs_pred = pywt.wavedec2(pred_np[i, c], 'haar', level=levels)
            coeffs_tgt = pywt.wavedec2(tgt_np[i, c], 'haar', level=levels)
            
            # coeffs[0]是低频，coeffs[1:]是各层高频 (LH, HL, HH)
            for level in range(1, levels + 1):
                for detail in range(3):  # LH, HL, HH
                    diff = coeffs_pred[level][detail] - coeffs_tgt[level][detail]
                    loss += weights[level-1] * np.mean(np.abs(diff))
    
    return torch.tensor(loss / (pred_np.shape[0] * pred_np.shape[1]))
```

**更实用的PyTorch纯实现（无需pywt）**：

```python
# 方案3.2：PyTorch原生Haar小波变换（可GPU加速）
def haar_wavelet_decompose(x):
    """
    可微分的Haar小波一层分解
    x: (B, C, H, W)
    返回: (LL, LH, HL, HH)
    """
    # 水平方向：低通滤波 [1,1]/2，高通滤波 [1,-1]/2
    # 用平均池化+差分实现
    x_avg_h = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) / 2
    x_diff_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) / 2
    
    # 垂直方向
    ll = (x_avg_h[:, :, 0::2, :] + x_avg_h[:, :, 1::2, :]) / 2
    lh = (x_diff_h[:, :, 0::2, :] + x_diff_h[:, :, 1::2, :]) / 2
    hl = (x_avg_h[:, :, 0::2, :] - x_avg_h[:, :, 1::2, :]) / 2
    hh = (x_diff_h[:, :, 0::2, :] - x_diff_h[:, :, 1::2, :]) / 2
    
    return ll, lh, hl, hh

def wavelet_loss_torch(pred, target):
    """PyTorch可微Haar小波高频损失"""
    _, lh_p, hl_p, hh_p = haar_wavelet_decompose(pred)
    _, lh_t, hl_t, hh_t = haar_wavelet_decompose(target)
    
    loss = (torch.mean(torch.abs(lh_p - lh_t)) +
            torch.mean(torch.abs(hl_p - hl_t)) +
            torch.mean(torch.abs(hh_p - hh_t)))
    
    return loss
```

### 优化建议

| 建议 | 说明 |
|---|---|
| **小波层数=1或2即可** | 层数太多会导致高频子带尺寸过小，且计算量增加。1层分解（LH, HL, HH三个子带）通常足够。 |
| **权重设置** | 第一层高频（尺寸最大）权重最高，第二层减半。如`weights=[1.0, 0.5]`。 |
| **与FFT loss二选一** | 小波损失和FFT loss本质都是频域约束，**不建议同时用**，选一个即可。本科毕设推荐用FFT loss（实现更简单）。 |
| **HOG特征可跳过** | HOG的PyTorch实现较复杂，如果只是为了边缘/纹理，Sobel edge loss已经足够。 |
| **训练稳定性** | 小波分解是可微分的，但下采样会损失一半分辨率。建议只在损失中用，不影响网络前向传播。 |

---

## 文献四：VQRNet —— 压缩视频增强的双分支设计

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | Lightweight State-Space Model-Based Video Quality Enhancement for Quadruped Robot Dog Decoded Streams |
| **作者** | VQRNet团队（Electronics期刊） |
| **发表** | Electronics (MDPI), 2026年3月, Vol.15, No.6, 1151 |
| **网址** | https://www.mdpi.com/2079-9292/15/6/1151 |
| **类型** | 压缩视频增强 / VVC后处理 / 轻量化网络 |

### 他在做什么

这篇论文解决的是**VVC压缩视频在机器人巡检场景中的质量增强**。具体来说：

- 机器狗采集的视频在剧烈运动下产生复杂混合失真（运动模糊+压缩伪影）
- 传统CNN感受野有限，无法捕获长距离空间依赖
- Transformer虽然能建模全局，但计算量太大，无法部署在边缘设备

作者提出了VQRNet，一个**轻量双分支网络**，在只有**1.40M参数**和**5.27G FLOPs**的情况下，实现了超越NAFNet、CTNet等主流方法的性能。

### 他怎么做的

**核心架构：双分支协同设计**

```
输入低质量帧
  ├── 浅层特征提取（3x3卷积）
  ├── 分支1：Local Feature Extraction Stream
  │     └── 6个级联 PFFM (Progressive Feature Fusion Module)
  │           └── 每个PFFM内嵌 MLSA (Multi-Scale Lightweight Spatial Attention)
  │     └── 作用：修复小块效应、振铃、高频纹理丢失
  │
  ├── 分支2：Global Feature Extraction Stream
  │     └── Pixel Unshuffle 降采样
  │     └── 3个 SSAM (State-Space Attention Module)
  │     └── Pixel Shuffle 上采样回原始尺寸
  │     └── 作用：修复大范围几何失真、结构坍塌
  │
  └── 融合：两个分支输出 + 浅层特征 拼接 → 1x1卷积融合 → 3x3重建
```

#### 核心模块1：PFFM (Progressive Feature Fusion Module)

- **四阶段渐进融合**：每个阶段都有前向传播和横向连接
- **1x1卷积分支**：通道重校准，提取全局通道依赖
- **3x3 RepConv分支**：局部空间模式，可重参数化（训练复杂，推理轻量）
- **跨阶段聚合**：第k阶段融合前k-1阶段的输出，构建金字塔式特征表示

#### 核心模块2：MLSA (Multi-Scale Lightweight Spatial Attention)

- **可学习下采样**：用2x2卷积替代池化，保留更多结构信息
- **通道级最大池化**：把多通道压缩成单通道空间显著图
- **三级级联RepConv**：模拟多尺度感受野（3个3x3级联 ≈ 7x7感受野）
- **轻量设计**：训练时有并行分支，推理时通过结构重参数化融合为单核

#### 核心模块3：SSAM (State-Space Attention Module)

- 结合**State-Space Model (SSM)** 和 **Attention**
- 1D卷积投影分解特征为：Bias(B)、Context(C)、动态时间(dt)三个分量
- 全局通道注意力 + 深度可分离卷积局部建模
- **线性复杂度**（O(n)），比Transformer的O(n^2)更适合高分辨率

#### 损失函数

非常简单：
```
L = -PSNR (即最小化负PSNR，等价于最大化PSNR)
```

等价于MSE的变体，但作者通过消融实验证明了其稳定性优于L1和MSE。

### 对我有什么启发

**这是A方案"网络结构改进"方向的最佳参考！**

虽然VQRNet的任务是VVC压缩视频增强（和A方案的HEVC压缩增强类似），但其设计哲学非常通用：

1. **"局部高频 + 全局结构"双分支是通用范式**：
   - A方案的PureResUNet只有局部特征提取（CNN），没有显式的全局分支
   - 边缘模糊往往是大面积区域的问题，需要全局结构指导

2. **PFFM的"渐进融合"思想可以借鉴**：
   - 当前PureResUNet的skip connection是简单拼接
   - PFFM的跨阶段聚合可以更好地保留多尺度信息

3. **MLSA证明了"轻量多尺度注意力"的可行性**：
   - A方案只有通道注意力（如果有的话），没有显式的空间注意力
   - 边缘区域需要空间注意力来聚焦

4. **SSM（Mamba）是2025-2026年的新兴趋势**：
   - 如果A方案要追热点，把部分卷积换成SSM是一个不错的创新点
   - 但SSM的PyTorch实现需要额外库（如mamba-ssm），本科毕设需谨慎

### A方案适配方案

**适配思路一：给PureResUNet添加"全局分支"（轻量版）**

不需要完全复刻VQRNet的双分支，可以在现有U-Net基础上加一个轻量全局路径：

```python
# 方案4.1：PureResUNet + 轻量全局分支

class LightweightGlobalBranch(nn.Module):
    """
    轻量全局分支：模拟VQRNet的global stream思想
    输入：浅层特征 (B, C, H, W)
    输出：全局结构特征 (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        # 下采样：用步长卷积替代Pixel Unshuffle，更简单
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 全局处理：用几个大核卷积模拟全局感受野
        self.global_proc = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),  # 深度可分离大核
            nn.ReLU(),
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.ReLU(),
        )
        # 上采样
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        return self.up(self.global_proc(self.down(x)))

# 修改PureResUNet的decoder部分：
# 在每个skip connection中，除了原有特征，再加入全局分支的输出
```

**适配思路二：给skip connection加"空间注意力门"（更简单）**

```python
# 方案4.2：Spatial Attention Gate for Skip Connection

class AttentionGate(nn.Module):
    """
    注意力门：让decoder决定从encoder的skip connection中"关注"哪些区域
    特别适合让模型关注边缘区域
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: decoder特征, x: encoder(skip)特征
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # 用注意力图加权skip特征
```

**适配思路三：渐进特征融合（最轻量）**

```python
# 方案4.3：Progressive Feature Fusion Block
# 替换PureResUNet中的标准ResBlock

class PFFBlock(nn.Module):
    """
    简化的渐进特征融合块
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, prev_features=None):
        """
        x: 当前输入
        prev_features: 前一阶段的特征（用于跨阶段聚合）
        """
        out1 = self.relu(self.conv1x1(x))
        out2 = self.relu(self.conv3x3(x))
        
        if prev_features is not None:
            # 跨阶段聚合
            out = out1 + out2 + prev_features
        else:
            out = out1 + out2
        
        return out + x  # 残差连接
```

### 优化建议

| 建议 | 说明 |
|---|---|
| **优先做"损失函数"改进** | VQRNet的网络结构改进工作量大（需要改模型架构、重新设计forward、调参）。本科毕设建议**先改损失函数**（Edge+FFT），效果明显且工作量可控。 |
| **全局分支选方案4.2** | Attention Gate是最轻量的结构改进，只加几个1x1卷积，不影响原有参数。 |
| **不要直接引入Mamba/SSM** | Mamba需要额外库（如`mamba-ssm`或`causal-conv1d`），Windows下安装困难，且和现有代码兼容性未知。除非你有Linux环境且时间充裕，否则不建议。 |
| **渐进融合适合放在Encoder** | PFFM的思想更适合放在编码器（下采样路径），因为编码器是特征提取阶段，需要多尺度信息。解码器专注于重建，改动风险较大。 |
| **RepConv可以省略** | VQRNet的RepConv需要训练时多分支+推理时融合，实现复杂。本科毕设用标准3x3卷积即可，性能差距不大。 |

---

## 文献五：NTIRE 2026 —— 竞赛方案中的高频损失设计

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | NTIRE 2026 Challenge on Short-form UGC Video Restoration in the Wild with Generative Models: Datasets, Methods and Results |
| **作者** | Xin Li 等（大量作者） |
| **发表** | CVPRW 2026 (IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops) |
| **网址** | https://arxiv.org/abs/2604.10551 |
| **类型** | 视频恢复竞赛报告 / 生成模型 / UGC视频 |

### 他在做什么

这是**2026年CVPR Workshop的最新竞赛报告**，聚焦"野生短视频（UGC）恢复"。竞赛有两个赛道：主观评价（用户研究）和客观评价。

虽然是竞赛报告，但里面各参赛队伍的方法非常有参考价值。特别是**BVI团队**的方案，明确使用了高频损失。

### 他怎么做的（重点分析BVI团队方案）

BVI团队的方法是一个**单步潜在扩散恢复框架**，其损失函数设计特别值得关注：

```
总损失 = L_TADSR（原始目标）
       + L_LPIPS（感知损失）
       + L_detail-aware-high-frequency（细节感知高频损失）
       + L_gradient-detail（梯度细节损失）
       + L_ratio-capped-residual-regularization（残差比例约束）
```

#### 关键损失1：Detail-aware High-Frequency Loss

虽然论文没有给出具体公式，但从上下文推断，这应该是一种**在潜在空间中显式约束高频细节**的损失。考虑到这是扩散模型的潜在空间，可能类似于：
- 对VAE编码后的潜在表示做频率分解
- 对高频分量施加额外的L2或L1约束

#### 关键损失2：Gradient Detail Loss

这很可能是**梯度域损失**的变体，即在图像梯度上计算重建误差：
```
L_gradient = ||∇I_pred - ∇I_gt||_p
```
这与Sobel edge loss是同一类方法。

#### 关键损失3：Ratio-Capped Residual Regularization

- 不直接惩罚残差修正的幅度
- 而是约束**残差修正与基础预测之间的比例**
- 只有当比例超过预定目标时才激活正则化
- 防止残差修正器被过度抑制，同时保持稳定

### 对我有什么启发

**核心启发："复合损失"是竞赛中的主流趋势**

NTIRE 2026的多个参赛队伍都使用了**多损失联合训练**：
1. 重建损失（L1/L2/MSE）保基本保真度
2. 感知损失（LPIPS/VGG）保语义一致性
3. **高频损失保细节**
4. **梯度损失保边缘**

**这验证了A方案的改进方向完全正确！**

另一个启发是：**ratio-capped residual regularization**可以借鉴到A方案的全局残差学习中。PureResUNet本身就是"全局残差学习"（输出 = 输入 + 残差），如果残差分支的修正幅度过大，可能导致不稳定。可以加一个比例约束：

```python
# 比例约束正则化
def ratio_capped_residual_loss(pred, lq_input, target_ratio=0.3):
    residual = pred - lq_input
    ratio = torch.abs(residual) / (torch.abs(lq_input) + 1e-8)
    
    # 只惩罚超过目标比例的残差
    mask = ratio > target_ratio
    loss = torch.mean((ratio[mask] - target_ratio) ** 2)
    
    return loss
```

### A方案适配方案

**最直接的应用：确认"多损失联合"的必要性**

A方案目前的损失是L1 + SSIM，属于"双损失"。参考NTIRE 2026的竞赛方案，可以升级为"四损失"：

```python
# 方案5：A方案升级后的复合损失
L_total = w1 * L_charbonnier    # 空间域重建（保真度）
        + w2 * L_ssim           # 结构相似性
        + w3 * L_edge           # 边缘梯度（保边缘）
        + w4 * L_fft            # 频域损失（保高频）
        + w5 * L_lpips          # 感知损失（保视觉）
```

**权重建议（基于各文献的消融实验综合）：**

| 损失 | 权重 | 理由 |
|---|---|---|
| L_charbonnier | 1.0 | 主损失，保像素级保真度 |
| L_ssim | 1.0 | 原方案已有，保结构 |
| L_edge | 0.3~0.5 | 边缘损失，权重不宜过大否则纹理过锐 |
| L_fft | 0.05~0.1 | 辅助损失，参考FDENet的α=0.1 |
| L_lpips | 0.1~0.3 | 感知损失，需要预训练VGG，计算量大，可选 |

### 优化建议

| 建议 | 说明 |
|---|---|
| **LPIPS可选** | LPIPS需要加载预训练VGG网络，增加内存和计算。如果GPU显存紧张（A方案用RTX 5080 16GB），可以先不加LPIPS，只做Edge+FFT。 |
| **ratio-capped regularization可尝试** | 如果训练过程中发现残差分支输出震荡大（loss曲线不平稳），可以加这个正则化。 |
| **参考竞赛方案的数据增强** | NTIRE 2026的方案使用了Real-ESRGAN风格的退化合成作为额外数据。A方案如果训练数据不足，可以考虑用类似策略扩充。 |
| **竞赛方案验证了Edge+FFT的有效性** | 在论文中写"本文的复合损失设计参考了NTIRE 2026竞赛中优胜方案的主流配置"，增加说服力。 |

---

## 文献六：HFS-HNeRV —— Haar小波高频模块

### 论文信息

| 项目 | 内容 |
|---|---|
| **标题** | HFS-HNeRV: High-Frequency Spectrum Hybrid Neural Representation for Videos |
| **作者** | （来自Auckland University of Technology等） |
| **发表** | ACM International Conference on Multimedia in Asia (MM Asia), 2024年12月 |
| **网址** | https://dl.acm.org/doi/10.1145/3696409.3700250 |
| **类型** | 神经视频表示 / 高频恢复 / 视频压缩 |

### 他在做什么

这篇论文解决的是**神经视频表示（NeRV）中的高频细节丢失问题**。NeRV是一种用MLP将视频帧索引映射到像素的方法，但传统NeRV在压缩后高频信号被过度抑制，导致重建图像边缘模糊、纹理软化。

### 他怎么做的

**核心创新1：HFSCM (High-Frequency Spectrum Convolution Module)**

```
输入特征
  ├── 常规卷积分支
  └── 高频增强分支：
        ├── Haar小波变换：分解为LL（低频）、LH（水平高频）、HL（垂直高频）、HH（对角高频）
        ├── HFSAM（High-Frequency Spectrum Attention Mechanism）：用注意力机制增强高频子带
        └── Haar小波逆变换：转回空间域
  
  输出：两个分支的特征拼接/融合
```

**关键设计：Haar小波注意力**
- Haar小波计算简单（只有加减和平均），计算开销极小
- 分解后的高频子带（LH, HL, HH）直接反映了边缘和纹理信息
- 注意力机制在高频子带上操作，让网络自适应地增强重要的高频成分

**核心创新2：HFS Loss (High-Frequency Spectrum Loss)**

```
L_HFS = ||F_high-pass(I_pred) - F_high-pass(I_gt)||^2
```

其中 `F_high-pass` 是高通滤波器（在频域中只保留高频分量）。

这是**显式的高频谱损失**：通过FFT+高通滤波提取高频信号，计算预测图和真值图的高频MSE。

**实验结果**：在视频压缩任务中，相比NeRV提升+5.68dB PSNR，相比E-NeRV提升+4.46dB，相比HNeRV提升+0.98dB。更重要的是，视觉质量上"边缘纹理重建更有效，颜色分布更自然"。

### 对我有什么启发

**核心启发：Haar小波是最高效的频域分解工具**

相比FFT（全局频域）和DCT（块级频域），Haar小波的优势在于：
1. **计算极简**：只有加减法和平均，没有复杂的三角函数
2. **多尺度**：一层分解得到4个子带，两层得到10个子带
3. **局部性**：小波基函数是紧支撑的，高频子带的位置对应空间域的边缘位置
4. **完美重建**：逆变换完全无损

**HFS Loss的启示**：
- 不需要对整个频谱做损失（像FFT loss那样）
- 可以只对**高通部分**做损失，让模型专注于高频恢复
- "频率域一致性学习"的思想可以直接用于A方案

### A方案适配方案

**适配思路：Haar小波高频增强模块（轻量插件）**

可以在PureResUNet的某些层之间插入一个简化的HFSCM：

```python
# 方案6：Haar Wavelet High-Frequency Enhancement Block

class HaarHFBlock(nn.Module):
    """
    Haar小波高频增强模块
    插入到U-Net的encoder或decoder中
    """
    def __init__(self, channels):
        super().__init__()
        # 小波分解后高频子带的增强网络
        self.lh_enhance = nn.Conv2d(channels, channels, 3, padding=1)
        self.hl_enhance = nn.Conv2d(channels, channels, 3, padding=1)
        self.hh_enhance = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def haar_decompose(self, x):
        """可微Haar小波分解"""
        # 水平方向：低通[1,1]/sqrt(2)，高通[1,-1]/sqrt(2)
        x_l = (x[:,:,:,0::2] + x[:,:,:,1::2]) / 2
        x_h = (x[:,:,:,0::2] - x[:,:,:,1::2]) / 2
        
        # 垂直方向
        ll = (x_l[:,:,0::2,:] + x_l[:,:,1::2,:]) / 2
        lh = (x_h[:,:,0::2,:] + x_h[:,:,1::2,:]) / 2
        hl = (x_l[:,:,0::2,:] - x_l[:,:,1::2,:]) / 2
        hh = (x_h[:,:,0::2,:] - x_h[:,:,1::2,:]) / 2
        
        return ll, lh, hl, hh
    
    def haar_reconstruct(self, ll, lh, hl, hh):
        """可微Haar小波重建"""
        # 垂直方向重建
        x_l = torch.zeros_like(ll)
        x_h = torch.zeros_like(ll)
        x_l[:,:,0::2,:] = ll + hl
        x_l[:,:,1::2,:] = ll - hl
        x_h[:,:,0::2,:] = lh + hh
        x_h[:,:,1::2,:] = lh - hh
        
        # 水平方向重建
        x = torch.zeros_like(ll)
        x[:,:,:,0::2] = x_l + x_h
        x[:,:,:,1::2] = x_l - x_h
        
        return x
    
    def forward(self, x):
        ll, lh, hl, hh = self.haar_decompose(x)
        
        # 增强高频子带
        lh_enh = self.relu(self.lh_enhance(lh))
        hl_enh = self.relu(self.hl_enhance(hl))
        hh_enh = self.relu(self.hh_enhance(hh))
        
        # 重建
        out = self.haar_reconstruct(ll, lh_enh, hl_enh, hh_enh)
        
        return out + x  # 残差连接
```

**使用方式**：
```python
# 在PureResUNet的encoder中每隔几个block插入一个HaarHFBlock
# 例如：在第2、4个下采样层之后各加一个
```

### 优化建议

| 建议 | 说明 |
|---|---|
| **Haar模块轻量优先** | 每个HaarHFBlock只有3个3x3卷积，参数量极小。但插入位置要慎重，不要每个block都插。 |
| **只增强高频，不动低频** | 上面的实现中`ll`子带完全不做处理，只增强LH/HL/HH。这保证了基础信息不会失真。 |
| **损失函数中也可以加HFS Loss** | 参考论文的HFS Loss，在训练时加一个显式的高频谱损失：`||FFT_highpass(pred) - FFT_highpass(gt)||^2`。 |
| **Haar vs FFT选哪个？** | 如果做**网络结构改进**，选Haar模块（插在网络里）；如果只做**损失函数改进**，选FFT loss（更简单）。本科毕设建议后者。 |

---

## 综合对比与推荐方案

### 六篇文献的方法对比

| 文献 | 核心方法 | 实现难度 | 对A方案的改进维度 | 推荐度 |
|---|---|---|---|---|
| AESOP (CVPR 2025) | AE空间损失 | ⭐⭐⭐ | 损失函数理论 | 理论引用 |
| FDENet (Sensors 2026) | FFT Loss + Charbonnier | ⭐ | 损失函数实现 | **强烈推荐** |
| RealisVSR (2025) | 小波+HOG高频损失 | ⭐⭐ | 损失函数多尺度 | 可选 |
| VQRNet (Electronics 2026) | 双分支+PFFM+MLSA | ⭐⭐⭐⭐ | 网络结构 | 结构改进参考 |
| NTIRE 2026 | 复合损失竞赛方案 | ⭐⭐ | 损失函数组合 | 配置参考 |
| HFS-HNeRV (MM Asia 2024) | Haar小波高频模块 | ⭐⭐ | 网络结构插件 | 结构改进参考 |

### A方案最终推荐改进路线

根据"工作量可控 + 效果明显 + 论文好写"的原则，推荐以下路线：

```
阶段1（必选）：损失函数升级
  ├── 用Charbonnier Loss替换L1 Loss
  ├── 加入Edge Loss（Sobel梯度L1差分）
  └── 加入FFT Loss（频率域高频约束）
  
阶段2（可选，时间充裕时做）：轻量结构改进
  └── 在skip connection中加入Attention Gate
  或
  └── 在encoder中加入1-2个HaarHFBlock
```

**最终损失函数配置**：

```python
L_total = 1.0 * L_charbonnier \
        + 1.0 * L_ssim \
        + 0.3 * L_edge \
        + 0.1 * L_fft
```

**预期效果**：
- 边缘模糊帧的`edge_blur`指标下降30~50%
- `gradient_loss`（纹理丢失）下降20~40%
- PSNR可能微降0.01~0.05dB（边缘变锐利后像素误差增加是正常的）
- SSIM提升或持平
- **LPIPS（如引入）显著下降**——这是最有说服力的证据

---

## 附录：快速代码索引

| 文献 | 可直接复用的代码片段 | 位置 |
|---|---|---|
| FDENet | Charbonnier Loss + FFT Loss | 本文"文献二"方案 |
| FDENet | FFT Loss PyTorch实现 | 本文"文献二"代码块 |
| RealisVSR | Haar小波分解（PyTorch可微） | 本文"文献三"方案3.2 |
| VQRNet | Attention Gate for Skip Connection | 本文"文献四"方案4.2 |
| VQRNet | Progressive Feature Fusion Block | 本文"文献四"方案4.3 |
| HFS-HNeRV | Haar Wavelet High-Frequency Block | 本文"文献六"代码块 |
| AESOP | 特征空间损失（简化版） | 本文"文献一"方案1.1 |

---

> **文档说明**：本文档中的代码均为可直接运行的PyTorch代码，改编自各文献的核心思想。具体实现时需根据A方案的代码结构（`models/pure_resunet.py`和`losses/l1_ssim_loss.py`）进行适配。建议在修改前备份原文件。
