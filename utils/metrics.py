#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# PSNR / SSIM 计算工具（验证/测试用）

import torch
import torch.nn.functional as F
import math


def calc_psnr(pred, target, max_val=1.0):
    # 计算整帧或 batch 的平均 PSNR
    # pred, target: torch.Tensor, shape (B, C, H, W) 或 (C, H, W), range [0, 1]
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val ** 2 / mse.item())
    return psnr


def calc_ssim(pred, target, window_size=11):
    # 计算整帧或 batch 的平均 SSIM（简化版，复用 losses 中的实现）
    # 为避免循环导入，这里内联实现
    from losses.l1_ssim_loss import _create_window, _ssim

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channel = pred.shape[1]
    window = _create_window(window_size, channel).to(pred.device)
    ssim_val = _ssim(pred, target, window, window_size, channel, size_average=True)
    return ssim_val.item()
