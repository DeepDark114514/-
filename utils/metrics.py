import torch
import torch.nn.functional as F
import math


def calc_psnr(pred, target, max_val=1.0):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val ** 2 / mse.item())
    return psnr


def calc_ssim(pred, target, window_size=11):
    from losses.l1_ssim_loss import _create_window, _ssim

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channel = pred.shape[1]
    window = _create_window(window_size, channel).to(pred.device)
    ssim_val = _ssim(pred, target, window, window_size, channel, size_average=True)
    return ssim_val.item()
