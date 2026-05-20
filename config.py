import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

CONFIG = {
    'model_type': 'A',
    'base_ch': 32,  # 32->12M参数，64->48M，5-15M比较合适就定了32

    'loss_type': 'L1_SSIM',
    'l1_weight': 1.0,
    'ssim_weight': 1.0,

    'optimizer': 'Adam',
    'lr': 1e-4,
    'weight_decay': 0,  # A方案没加，试过1e-4效果反而差，先保持0
    'scheduler': 'StepLR',
    'step_size': 30,
    'gamma': 0.1,
    'epochs': 100,
    'batch_size': 32,  # 5080 16G显存，bs=32大概占7-8G，再多会炸
    'patch_size': 256,
    'num_workers': 6,
    'pin_memory': False,  # Windows开这个DataLoader容易崩，Linux可以开
    'persistent_workers': True,
    'seed': 42,
    'grad_accum_steps': 1,

    'early_stop': True,
    'early_stop_patience': 6,  # val_interval=5，也就是30个epoch不升就停，试过10太久了
    'early_stop_min_delta': 0.0,
    'early_stop_monitor': 'val_psnr',
    'early_stop_mode': 'max',

    'val_interval': 5,  # 每5轮验证一次，太频繁浪费时间
    'save_interval': 10,
    'clip_grad_norm': 1.0,

    'device': 'cuda',
    'amp': False,  # FP16在这个Pre-Activation结构下会NaN，只能FP32
    'pred_clamp': True,
    'dataset_root': str(ROOT_DIR / 'MFQEv2_processed'),
    'root': str(ROOT_DIR / 'MFQEv2_processed'),

    'train_list': 'train_list.txt',
    'val_list': 'val_list.txt',
    'test_list': 'test_list.txt',
    'qp': 32,  # 验证/测试基准QP，训练时A和B都用qp_list
    'qp_list': [22, 32, 42],
}
