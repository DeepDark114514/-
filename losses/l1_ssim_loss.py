#  南京信息工程大学22级信安1班 202283290014
# 2026.5.12
# A/B 共用损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window_1d(size, sigma):
    # 生成一维高斯核
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size, channel):
    # 生成二维高斯核 (C, 1, window_size, window_size)
    _1d_window = _gaussian_window_1d(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 计算单尺度 SSIM
    # img1, img2: (B, C, H, W), 范围 [0, 1]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class L1SSIMLoss(nn.Module):
    # L1 + SSIM
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, window_size=11):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.channel = 3
        self.window = None

    def forward(self, pred, target, return_components=False):
        if self.window is None or self.window.device != pred.device:
            self.window = _create_window(self.window_size, self.channel).to(pred.device)

        l1 = F.l1_loss(pred, target)
        ssim_val = _ssim(pred, target, self.window, self.window_size, self.channel, size_average=True)
        ssim = 1.0 - ssim_val  # SSIM范围[0,1]，1-ssim作为loss
        total = self.l1_weight * l1 + self.ssim_weight * ssim

        if return_components:
            return total, l1.item(), ssim.item(), ssim_val.item()
        return total
