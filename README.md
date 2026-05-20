# MFQEv2 视频压缩伪影去除

202283290014

A/B 对比实验。A 是基线 PureResUNet，B 加了 DegFiLM 退化感知模块。

数据集 MFQEv2，从 HEVC 参考数据集来的，108 个视频序列。用 HM-16.20 压的，训只有 QP22/32/42，测有 QP22/27/32/37/42。

装依赖：
```
pip install -r requirements.txt
```

训练：
```
python train.py -m A --epochs 100
python train.py -m B --epochs 100
```

参数在 config.py 里改。

评估：
```
python scripts/cross_qp_eval.py --model_path logs/xxx/best_model.pth --model_type A --qp_list 22 27 32 37 42
python scripts/eval_baseline.py
python scripts/eval_model_A.py --exp_dir logs/xxx
```

注意：
- AMP 关的，FP16 下 Pre-Activation ResBlock 会 NaN
- Windows 上 pin_memory 关掉
- cuDNN benchmark 开了
