# MFQEv2 视频压缩伪影去除

202283290014

A/B 对比实验。A 是基线 PureResUNet，B 加了 DegFiLM 退化感知模块。

---

## 数据集

本项目用 108 个 YUV 视频序列，来自公开的 HEVC 标准测试集、Xiph、Netflix Open Content 等。需自行下载原始 YUV（8bit, YUV420p），然后用 HM-16.20 编码得到 LQ。

**HM-16.20 下载**：https://vcgit.hhi.fraunhofer.de/jct-vc/HM/-/tags/ HM-16.20

编码命令示例（以 QP32 为例）：
```bash
TAppEncoder.exe -c configs/encoder_ldp_template.cfg -c configs/test_seq_BasketballPass.cfg -q 32
```
训练集压 QP22/32/42，测试集压 QP22/27/32/37/42。

**目录结构**：
```
MFQEv2_processed/
├── gt/
│   ├── train/          # 原始 YUV
│   ├── val/
│   └── test/
└── compressed/
    ├── train/          # HM 压缩后的 YUV，命名如 xxx_qp32.yuv
    ├── val/
    └── test/
```
训练/验证/测试的序列名单见 `MFQEv2_processed/train_list.txt` 等。

---

## 环境

```bash
pip install -r requirements.txt
```

Windows + PyTorch，显卡 RTX 5080 16G。

---

## 训练

```bash
python train.py -m A --epochs 100   # A 方案，固定 QP32
python train.py -m B --epochs 100   # B 方案，混合 QP22/32/42
```

参数在 `config.py` 里改，比如 batch_size、lr。

---

## 评估

```bash
python scripts/cross_qp_eval.py --model_path logs/xxx/best_model.pth --model_type A --qp_list 22 27 32 37 42
python scripts/eval_baseline.py
python scripts/eval_model_A.py --exp_dir logs/xxx
```

---

## 注意

- **AMP 关**：Pre-Activation ResBlock 在 FP16 下数值会漂移，loss 变 NaN，只能 FP32 训。
- **Windows 上 pin_memory 关掉**：Windows 多进程机制跟 Linux 不一样，开 true DataLoader 容易死锁报错。
- **cuDNN benchmark 开**：输入尺寸固定（256x256 patch）时，PyTorch 会自动选最快的卷积算法，训练速度能快一点。
