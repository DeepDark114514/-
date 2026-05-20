# MFQEv2 视频压缩伪影去除

202283290014

A/B 对比实验。A 是基线 PureResUNet，B 加了 DegFiLM 退化感知模块。

数据集是 108 个 YUV 视频序列，自己从 HEVC 标准测试集、Xiph、Netflix Open Content 那些公开数据集里收集的原始 YUV，然后用 ffmpeg + libx265 压的。训练集压了 QP22/32/42 三档，测试集压了 QP22/27/32/37/42 五档。

没用 HM 参考软件，因为 ffmpeg 快很多，压 108 个视频省大量时间。而且深度学习训练对编码器细节不敏感，libx265 和 HM 压出来的伪影分布足够接近，对最终模型性能影响不大。

编码命令示例（QP32）：
```
ffmpeg -f rawvideo -pix_fmt yuv420p -s 416x240 -r 50 -i input.yuv -c:v libx265 -x265-params "qp=32" -f rawvideo -pix_fmt yuv420p output_qp32.yuv
```
训练集压 22/32/42，测试集压 22/27/32/37/42。

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
```
跑完会在 result/cross_qp/ 里输出 JSON、CSV 和几张退化曲线图（PSNR/SSIM 退化曲线、增益对比图、增益波动图）。这就是论文里跨 QP 泛化性能的数据来源。

cross_qp_eval 本身就带 baseline 计算，每个 QP 都会算 LQ 输入的 PSNR/SSIM 当基准，所以不需要额外跑 baseline 脚本。

效率测试（可选，论文效率章节的数据来源）：
```
python scripts/benchmark_ab_forward.py      # 测 A/B 推理速度和峰值显存
python scripts/benchmark_ab_trainstep.py    # 测 A/B 训练耗时拆解
```

result/ 里还有一些历史结果和可视化图（比如 baseline/、ab_visualization_v2/），是之前用旧脚本跑的，现在保留作参考。如果要复现那些可视化对比图，得用本地的辅助脚本，不在仓库里。

几个坑：

- AMP 关的。Pre-Activation ResBlock 在 FP16 下数值会漂移，loss 变 NaN，只能 FP32 训。
- Windows 上 pin_memory 关掉。Windows 多进程机制跟 Linux 不一样，开 true DataLoader 容易死锁报错。
- cuDNN benchmark 开了。输入尺寸固定（256x256 patch）时 PyTorch 会自动选最快的卷积算法，训练速度能快一点。

项目结构没什么特别的，模型在 models/，数据加载在 datasets/，训练入口就一个 train.py。
