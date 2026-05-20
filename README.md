# MFQEv2 视频压缩伪影去除

202283290014

A/B 对比实验。A 是基线 PureResUNet，B 加了 DegFiLM 退化感知模块。

数据集是 108 个 YUV 视频序列，自己从 HEVC 标准测试集、Xiph、Netflix Open Content 那些公开数据集里收集的原始 YUV，然后用 HM-16.20 编码器压的。训练集压了 QP22/32/42 三档，测试集压了 QP22/27/32/37/42 五档。

HM-16.20 下载地址：https://vcgit.hhi.fraunhofer.de/jct-vc/HM/-/tags/ 找 HM-16.20

编码命令示例（QP32）：
```
TAppEncoder.exe -c configs/encoder_ldp_template.cfg -c configs/test_seq_BasketballPass.cfg -q 32
```
训练集压 22/32/42，测试集压 22/27/32/37/42。LDP 配置， GOP=4，CQP 模式。

压完之后的目录结构要像这样，代码才能读到：
```
MFQEv2_processed/
  gt/train/              原始 YUV
  gt/val/
  gt/test/
  compressed/train/      压缩后的 YUV，比如 xxx_qp32.yuv
  compressed/val/
  compressed/test/
```
序列名单在项目里已经有了，train_list.txt、val_list.txt、test_list.txt。

装依赖：
```
pip install -r requirements.txt
```

训练：
```
python train.py -m A --epochs 100
python train.py -m B --epochs 100
```
A 训练用 QP32，B 训练混着 QP22/32/42，因为 B 要做退化感知。其他参数在 config.py 里改，比如 batch_size、lr。

评估：
```
python scripts/cross_qp_eval.py --model_path logs/xxx/best_model.pth --model_type A --qp_list 22 27 32 37 42
python scripts/eval_baseline.py
python scripts/eval_model_A.py --exp_dir logs/xxx
```

cross_qp_eval 是核心评估脚本，跑跨 QP 泛化。eval_baseline 算压缩后 LQ 的 baseline 指标，论文里提升多少 dB 就靠它当基准。eval_model_A 是 test set 全面测评，逐帧算 PSNR/SSIM，结果存 JSON。

几个坑：

- AMP 关的。Pre-Activation ResBlock 在 FP16 下数值会漂移，loss 变 NaN，只能 FP32 训。
- Windows 上 pin_memory 关掉。Windows 多进程机制跟 Linux 不一样，开 true DataLoader 容易死锁报错。
- cuDNN benchmark 开了。输入尺寸固定（256x256 patch）时 PyTorch 会自动选最快的卷积算法，训练速度能快一点。

项目结构没什么特别的，模型在 models/，数据加载在 datasets/，训练入口就一个 train.py。结果和可视化在 result/ 里。
