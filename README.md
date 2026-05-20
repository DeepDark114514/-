# MFQEv2 视频压缩伪影去除

202283290014

A/B 方案对比实验：基线 PureResUNet vs 退化感知 DegFiLM-ResUNet。

## 数据集

**MFQEv2**（Multi-format Video Quality Enhancement）

- 来源：CVPR 2019 工作，包含 108 个未压缩视频序列
- 训练集：98 序列 | 验证集：10 序列 | 测试集：18 序列
- 分辨率覆盖：416x240 ~ 2560x1600

### 数据预处理

用 HM-16.20 编码器生成压缩序列：

```bash
# 生成 QP22/27/32/37/42 的压缩 YUV
python scripts/preprocess_multi_qp.py --input_dir MFQEv2_raw --output_dir MFQEv2_processed
```

处理流程：
- 原始 YUV -> HM 编码（LDP 配置，指定 QP）-> 解码回 YUV
- 输出结构：`compressed/{split}/seq_name_qp{qp}.yuv` 与 `gt/{split}/seq_name.yuv`

## 本地配置

测试环境：
- GPU：NVIDIA GeForce RTX 5080 16GB
- CPU：AMD Ryzen 7 9800X3D
- OS：Windows 11
- Python 3.10 + PyTorch 2.x

安装依赖：

```bash
pip install -r requirements.txt
```

## 训练

### A 方案（基线）

```bash
python train.py --model_type A --base_ch 32 --epochs 100
```

### B 方案（退化感知）

```bash
python train_b.py --use_film 1 --base_ch 32 --epochs 100
```

B 方案在 QP32 上训练，混合 QP22/32/42 数据增强泛化能力。

训练配置（config.py 关键项）：
- loss：L1 + SSIM，权重 1.0+1.0
- optimizer：Adam，lr=1e-4
- batch_size：32，patch_size：256
- early_stop：patience=6，监控 val_psnr
- 精度：FP32（AMP 关闭，避免 Pre-Activation ResBlock 数值漂移）

## 评估

### 跨 QP 泛化评估

```bash
# A 方案
python scripts/cross_qp_eval.py --model_path checkpoints/best_model.pth --model_type A --qp_list 22 27 32 37 42 --split test

# B 方案
python scripts/cross_qp_eval.py --model_path logs/B_20260519_121523/best_model.pth --model_type B --qp_list 22 27 32 37 42 --split test --out_dir result/cross_qp/B_20260519_121523
```

### A/B 对比汇总

```bash
python scripts/compare_ab_cross_qp.py
```

输出对比图表和 CSV 到 `result/cross_qp/B_20260519_121523/`。

### 效率基准

```bash
python scripts/benchmark_ab_forward.py
python scripts/benchmark_ab_trainstep.py
```

## 项目结构

```
.
├── config.py              # 训练配置
├── train.py               # A 方案训练入口
├── train_b.py             # B 方案训练入口
├── models/
│   ├── base_unet.py       # U-Net 骨架
│   ├── pure_resunet.py    # A 方案
│   ├── degfilm_resunet.py # B 方案
│   └── deg_film_blocks.py # DegEstimator + FiLM
├── datasets/              # 数据加载与 YUV IO
├── losses/                # L1 + SSIM
├── utils/                 # 早停、指标、优先级
├── scripts/               # 评估、对比、可视化
├── docs/                  # 技术文档
└── result/                # 评估结果
```

## 核心结论

B 方案以 2.3% 参数增量，在低压缩 QP（22/27）下实现 6~11% 相对增益，高 QP（42）与 A 基本持平。说明通道级 FiLM 的调节粒度在轻度失真域有效，重度失真域已接近架构瓶颈。
