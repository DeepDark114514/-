import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size, channel):
    _1d_window = _gaussian_window_1d(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
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


# L1 + SSIM 组合损失，没加感知损失因为显存不够
# L1比MSE对异常值不敏感，SSIM保结构，1:1配比自己试的
class L1SSIMLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, window_size=11):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size  # 11是经典设置，sigma=1.5
        self.channel = 3
        self.window = None

    def forward(self, pred, target, return_components=False):
        if self.window is None or self.window.device != pred.device:
            self.window = _create_window(self.window_size, self.channel).to(pred.device)

        l1 = F.l1_loss(pred, target)
        ssim_val = _ssim(pred, target, self.window, self.window_size, self.channel, size_average=True)
        ssim_loss = 1.0 - ssim_val
        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss

        if return_components:
            return total, l1.item(), ssim_loss.item(), ssim_val.item()
        return total
