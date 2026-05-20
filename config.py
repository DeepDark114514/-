#  南京信息工程大学22级信安1班 202283290014
# 2026.5.10
# A 方案训练框架配置
# 当前仅支持 PureResUNet（纯残差U-Net + 全局残差学习）

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.resolve()

CONFIG = {
    # 网络配置
    'model_type': 'A',          # A=PureResUNet(基线), B=DegFiLMResUNet(退化感知FiLM)
                                # 命令行切换: python train.py -m B
    'base_ch': 32,             # 基础通道数
                                # A方案: base_ch=32 时约 12.2M 参数
                                # B方案: base_ch=32 时约 12.3M 参数 (+0.8%, DegEstimator仅+0.05M)
                                #       base_ch=64 时约 48M，如需严格 5-15M 请保持 32

    # 损失函数
    'loss_type': 'L1_SSIM',
    'l1_weight': 1.0,          # L1 损失权重
    'ssim_weight': 1.0,        # SSIM 损失权重

    # 训练参数
    'optimizer': 'Adam',
    'lr': 1e-4,
    'weight_decay': 0,         # A方案先不加正则
    'scheduler': 'StepLR',
    'step_size': 30,
    'gamma': 0.1,
    'epochs': 100,
    'batch_size': 32,           # RTX 5080 16GB 全力输出
    'patch_size': 256,
    'num_workers': 6,            # 9800X3D 16线程，Windows 下 6 是稳定与速度的最优平衡点
    'pin_memory': False,        # Windows 保持 False，避免 pin_memory 线程 CUDA 错误
    'persistent_workers': True,  # Windows: 保持 worker 进程存活，避免每 epoch 重建开销
    'seed': 42,
    'grad_accum_steps': 1,       # 梯度累积步数，1=关闭累积，bs=16 时等效 batch_size=16

    # 早停机制
    'early_stop': True,
    'early_stop_patience': 6,        # 连续6次验证PSNR不提升则停（val_interval=5，即30个epoch）
    'early_stop_min_delta': 0.0,     # 提升小于此值视为无提升
    'early_stop_monitor': 'val_psnr', # 监控指标
    'early_stop_mode': 'max',         # PSNR越大越好

    # 验证/保存
    'val_interval': 5,           # 每5个epoch验证一次
    'save_interval': 10,         # 每10epoch存checkpoint
    'clip_grad_norm': 1.0,       # 梯度裁剪（在累积后执行）

    # 系统
    'device': 'cuda',
    'amp': False,                # 混合精度（AMP）。本架构（Pre-Activation ResBlock + 无界输出）
                                # 在 AMP (FP16) 下存在数值漂移，导致训练发散/loss=NaN。
                                # FP32 训练稳定，且显存仅 ~2.8GB (bs=8)，故默认关闭。
    'pred_clamp': True,          # 训练时对网络输出做 clamp(0,1) 再算 loss。
                                # 避免模型学到非法像素值，确保训练目标和验证指标一致。
    'dataset_root': str(ROOT_DIR / 'MFQEv2_processed'),
    'root': str(ROOT_DIR / 'MFQEv2_processed'),  # 兼容 data/build_dataloader 接口

    # 数据集文件列表
    'train_list': 'train_list.txt',
    'val_list': 'val_list.txt',
    'test_list': 'test_list.txt',
    'qp': 32,                    # 验证/测试基准QP
    'qp_list': [22, 32, 42],     # 训练时混合的多QP列表，每个样本随机抽取一个QP
}
