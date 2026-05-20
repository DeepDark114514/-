import torch
import torch.nn as nn


# 退化估计器，轻量CNN。3层够了，太深参数涨太多
class DegEstimator(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        feat = self.net(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


# FiLM: 用embed生成gamma/beta，给特征图做仿射变换
class FiLM(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)  # 初始化为0，训练初期gamma=0,beta=0，FiLM不起作用，稳定

    def forward(self, embed):
        return self.fc(embed)


# 带FiLM的ResBlock，bottleneck和decoder里用
class FiLMResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x, film_params=None):
        out = self.body(x)
        if film_params is not None:
            B, C = out.size(0), out.size(1)
            gamma = film_params[:, :C].view(B, C, 1, 1)
            beta = film_params[:, C:].view(B, C, 1, 1)
            out = (1 + gamma) * out + beta  # (1+gamma)而不是gamma，这样gamma=0时是恒等映射
        return x + out


__all__ = ['DegEstimator', 'FiLM', 'FiLMResBlock']
