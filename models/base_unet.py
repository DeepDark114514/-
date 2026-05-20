#  南京信息工程大学22级信安1班 202283290014
# 2026.5.15
# Base UNet 骨架 + 公共组件
# 供 A/B 方案继承/复用，减少代码复制
# 包含:
    # - ResBlock: Pre-Activation 残差块 (A/B 通用)
    # - BaseUNet: UNet 编码器-解码器骨架

import torch
import torch.nn as nn


# 公共组件

class ResBlock(nn.Module):
    # Pre-Activation ResBlock
    # forward 支持 side_input 接口以兼容不同调用方式
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x, side_input=None):
        # side_input 为条件化模块预留，标准 ResBlock 始终忽略
        out = self.body(x)
        return x + out


# Base UNet 骨架

class BaseUNet(nn.Module):
    # UNet 基础骨架
    # 编码器: 4层下采样，每层 2xResBlock + stride=2 Conv
            # 通道: base_ch -> 2*base_ch -> 4*base_ch -> 8*base_ch
    # 瓶颈层: 2xResBlock
    # 解码器: 4层上采样，每层 bilinear + 3x3Conv降通道 + SkipConcat + 2xResBlock
    # 输出头: 3x3 Conv -> 3通道，无激活函数
    # Skip Connection 的融合方式由 _fuse_skip() 方法控制，
    # 子类可 override 以实现不同的 skip 策略。
    def __init__(self, base_ch=64, in_channels=3, out_channels=3):
        super().__init__()
        self.base_ch = base_ch
        # 编码器每层的通道数（ResBlock 所在通道）
        enc_chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)

        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(4):
            enc_in_ch = enc_chs[i]
            down_out_ch = enc_chs[i + 1] if i < 3 else enc_chs[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(enc_in_ch),
                ResBlock(enc_in_ch),
            ]))
            self.down_blocks.append(nn.Conv2d(enc_in_ch, down_out_ch, 3,
                                             stride=2, padding=1, bias=False))

        # 瓶颈层
        self.bottleneck = nn.ModuleList([
            ResBlock(enc_chs[-1]),
            ResBlock(enc_chs[-1]),
        ])

        # 解码器
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_chs = [enc_chs[-1], enc_chs[2], enc_chs[1], enc_chs[0]]  # 从深到浅
        skip_chs = list(enc_chs)  # [base_ch, 2*base_ch, 4*base_ch, 8*base_ch]

        for i in range(4):
            in_ch = dec_chs[i - 1] if i > 0 else enc_chs[-1]
            out_ch = dec_chs[i]
            skip_ch = skip_chs[3 - i]  # 从深到浅取 skip

            # 上采样 + 降通道 conv
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            ))
            # 融合 conv + 2x ResBlock
            self.dec_blocks.append(nn.ModuleList([
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                ResBlock(out_ch),
                ResBlock(out_ch),
            ]))

        # 输出头
        self.out_conv = nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=False)

    def _fuse_skip(self, dec_feat, skip_feat, level):
        # 融合上采样后的解码器特征和 skip 特征。
        # Args:
            # dec_feat: 解码器上采样后的特征 (B, out_ch, H, W)
            # skip_feat: 编码器对应层的 skip 特征 (B, skip_ch, H, W)
            # level: int, 0-3, 0 为最深层，3 为最浅层
        # 子类可 override 此方法以实现不同的 skip 连接策略。
        x = torch.cat([dec_feat, skip_feat], dim=1)
        conv, res1, res2 = self.dec_blocks[level]
        x = conv(x)
        x = res1(x)
        x = res2(x)
        return x

    def forward(self, x):
        identity = x  # 全局残差：保存输入

        # 初始特征
        x = self.init_conv(x)

        # 编码器
        skips = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            skips.append(x)  # 保存 skip 特征 [e1, e2, e3, e4]
            x = self.down_blocks[i](x)

        # 瓶颈
        for res in self.bottleneck:
            x = res(x)

        # 解码器
        for i in range(4):
            x = self.up_blocks[i](x)
            skip = skips[3 - i]  # 从深到浅取 skip
            x = self._fuse_skip(x, skip, i)

        # 输出：残差学习，网络只学差异
        x = self.out_conv(x)
        return x + identity

    def get_encoder_features(self, x):
        # 显式返回编码器特征列表 [e1, e2, e3, e4]，供扩展方案使用
        x = self.init_conv(x)
        feats = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            feats.append(x)
            x = self.down_blocks[i](x)
        return feats, x
