# MFQEv2 视频压缩伪影去除

202283290014

A/B 方案对比实验。A 是基线 PureResUNet，B 加了 DegFiLM 退化感知模块。

数据集是 MFQEv2，从 HEVC 参考数据集来的，一共 108 个视频序列。自己用 HM-16.20 编码器压的，训练集只有 QP22/32/42 三档，测试集有 QP22/27/32/37/42 五档。

预处理：

```bash
python scripts/preprocess_multi_qp.py --input_dir MFQEv2_raw --output_dir MFQEv2_processed
```

环境就是 Windows + PyTorch，依赖装一下：

```bash
pip install -r requirements.txt
```

训练：

```bash
python train.py --model_type A --epochs 100   # A 方案，固定 QP32
python train.py --model_type B --epochs 100   # B 方案，混合 QP22/32/42
```

其他参数在 config.py 里改，比如 batch_size、lr 这些。

评估：

```bash
python scripts/cross_qp_eval.py --model_path logs/xxx/best_model.pth --model_type A --qp_list 22 27 32 37 42
python scripts/compare_ab_cross_qp.py
```

几个坑：

- AMP 默认关的。Pre-Activation ResBlock 在 FP16 下会 NaN，只能 FP32 训。
- Windows 上 pin_memory 也关掉了，不然 DataLoader 容易报错。
- cuDNN benchmark 开了，输入尺寸固定的时候有加速。

项目结构没什么特别的，模型在 models/，数据加载在 datasets/，训练入口就一个 train.py。结果和可视化在 result/ 里。
