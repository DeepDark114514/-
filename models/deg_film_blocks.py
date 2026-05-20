#  南京信息工程大学22级信安1班 202283290014
# 2026.5.17
# DegFiLM 核心模块
# 供 B方案 (DegFiLM-ResUNet) 使用
# 包含:
    # - DegEstimator: 轻量CNN旁路，提取全局退化嵌入
    # - FiLM: 从退化嵌入生成 gamma/beta，做逐通道仿射变换
    # - FiLMResBlock: 支持 FiLM 注入的 Pre-Activation ResBlock
# 设计约束:
    # - 总参数量增加 < 2% (相对于 A方案 ~12M)
    # - 端到端联合训练，无需预训练
    # - FiLM 初始化接近 0，训练初期近似恒等映射

import torch
import torch.nn as nn


class DegEstimator(nn.Module):
    # 轻量退化估计器
    # 从输入帧提取全局退化嵌入向量，参数量 ~0.05M
    # 结构: 3层 stride=2 下采样 + GAP + FC
          # 空间分辨率: 1 -> 1/2 -> 1/4 -> 1/8
          # 通道数: 32 -> 64 -> 128 (容量增大版)
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


class FiLM(nn.Module):
    # FiLM 生成器
    # 从退化嵌入向量生成逐通道的 gamma 和 beta
    # Args:
        # embed_dim: 退化嵌入维度
        # out_channels: 目标特征通道数
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_channels * 2)
        # 初始化接近 0，保证训练初期 FiLM 输出 gamma≈0, beta≈0
        # 此时 FiLMResBlock 中 (1+gamma)·body_out + beta ≈ body_out，近似标准 ResBlock
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, embed):
        # Args:
            # embed: (B, embed_dim)
        # Returns:
            # (B, 2*out_channels) —— 前一半为 gamma，后一半为 beta
        return self.fc(embed)


class FiLMResBlock(nn.Module):
    # 支持 FiLM 注入的 Pre-Activation ResBlock
    # 与 A方案 ResBlock 的区别:
        # - forward 支持可选的 film_params
        # - 在 body 输出后、残差相加前，插入 (1 + gamma) * body_out + beta
    # 当 film_params=None 时，行为与标准 ResBlock 完全一致。
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
            out = (1 + gamma) * out + beta
        return x + out


__all__ = ['DegEstimator', 'FiLM', 'FiLMResBlock']
