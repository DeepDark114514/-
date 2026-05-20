# QQ导师汇报草稿（v2学术严谨版）

---

**导师好，我汇报一下实验进度：**

我写了一个综合缺陷评估脚本来寻找论文的缺陷切入点。我参考了学术界目前的几个经典指标和方法，不是凭空设计的：

**1. 边缘/纹理评估**
参考了 **Li & Bovik, "Content-weighted video quality assessment using a three-component image model", Journal of Electronic Imaging, 2010**，论文PDF地址：
https://live.ece.utexas.edu/publications/2010/li_jei_jan10.pdf

这篇论文提出了用 **Sobel梯度把图像分成边缘区、纹理区和平滑区** 三个部分，然后在不同区域分别评估质量。我的"边缘模糊"指标就是基于这个三分区模型，在**边缘像素**上计算MAE；"纹理丢失"指标是在**纹理区**计算MAE。阈值用的是原文标准的 TH1=0.12*gmax, TH2=0.06*gmax。

**2. 高频损失评估**
参考了 **MDPI Applied Sciences 2024, "Research on a Multidimensional Digital Printing Image Quality Evaluation Method Based on MLP Neural Network Regression"**，论文地址：
https://www.mdpi.com/2076-3417/14/14/5986

这篇论文提出了用 **FFT计算高频能量比（HF Ratio）** 来评估图像细节丰富度。我的高频损失指标就是通过 **2D FFT** 计算模型输出相对于真值的高频能量衰减。

**3. 过度平滑评估**
参考了 **Xu et al., "Uncovering the Over-smoothing Challenge in Image Super-Resolution: Entropy-based Quantification and Contrastive Optimization", 2022（后发表于IEEE TPAMI 2024）**，论文地址：
https://arxiv.org/abs/2201.01034

这篇论文从**数据熵**的角度分析了PSNR导向模型为什么会过平滑，提出了用局部统计量来量化过平滑程度。我的"局部标准差MSE"指标就是参考了这个思路，用 **7x7窗口的局部标准差图** 来衡量细节密度差异。

**4. 块效应评估（虽然最终证明不重要）**
方法上参考了 **Wang et al., "Blind measurement of blocking artifacts in images", IEEE ICIP 2000**，论文地址：
https://doi.org/10.1109/ICIP.2000.899622

---

了解了这些指标的评估原理之后，针对我的**A方案（PureResUNet）**在 **MFQEv2测试集（18个序列、7980帧）** 上进行了逐帧分析，重点在 **边缘保持、纹理细节保持、高频信息保持和局部平滑度** 这四个方向上做了定量评估。阈值不是拍脑袋定的，是基于全量数据的 **95th percentile** 自适应确定的。

---

**最后发现：**

- 最主要的问题是 **边缘模糊**，38个代表性问题帧中有 **11帧（28.9%）** 存在严重的边缘保持误差；
- 其次是 **纹理丢失**，有 **10帧（26.3%）**；
- 这两个问题加起来占了 **55%以上**，说明模型的核心缺陷是在重建高频细节和保持边缘锐利度方面做得不够好；
- 另外还发现了一个现象：有 **5帧** 存在 **"PSNR-SSIM分歧"**，就是PSNR涨了很多但感知质量提升不明显，说明PSNR指标本身也有局限性；
- 而 **块效应、振铃、伪影** 这些指标在全量数据中**几乎没有增加**（块效应p95≈0，振铃全部为负），说明A模型不会引入这些新的artifacts，这些方向不用考虑了。

---

**所以我的结论是：**

把论文的改进切入点定在 **"边缘模糊 + 过度平滑"** 这个方向上，下一步准备尝试在loss中加入 **Edge Loss（Sobel边缘约束）+ FFT Loss（频域高频约束）** 来改善。这个方向有明确的文献支撑（Xu 2022分析了L1损失必然导致过平滑的机理），改进路径也比较清晰。

---

**生成的分析文件位置：**
- 7980帧完整指标：`result/A/per_frame_metrics_v2.json`
- 38帧问题帧汇总：`result/A/problem_frames_summary_v2.json`
- 可视化对比图：`result/A/visualizations/problem_v2/*.png`
