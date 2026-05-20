#  南京信息工程大学22级信安1班 202283290014
# 2026.5.17
# B方案：DegFiLM-ResUNet
# 盲QP视频压缩伪影去除网络
# 核心思想 (PromptCIR + FiLM):
    # - 不预测显式QF，而是学习隐式退化嵌入
    # - 用 FiLM 在残差块内部做逐通道仿射变换，自适应调节残差出力大小
    # - 端到端联合训练，DegEstimator 与主干一起优化，无需预训练
# 插入位置（只插深层，不插浅层）:
    # - Encoder:   不插（浅层负责边缘/颜色，不需要退化感知）
    # - Bottleneck: 插入 2x FiLM（感受野最大，需退化指导全局语义）
    # - Decoder level 0,1,2: 插入（重建阶段需退化指导恢复力度）
    # - Decoder level 3:    不插（最浅层最高分辨率，只负责细节精修）
    # - 不搞显式DRL预训练+冻结
    # - 不搞多级Stage+分层终止
    # - 不搞STDA空间-通道联合调制
    # - 走轻量路线：端到端训练，只调通道统计量，静态深度

import torch
import torch.nn as nn

from .base_unet import BaseUNet, ResBlock
from .deg_film_blocks import DegEstimator, FiLM, FiLMResBlock


class DegFiLMResUNet(BaseUNet):
    # B方案：DegFiLM-ResUNet
    # 继承 BaseUNet 骨架，仅改动 Bottleneck 和 Decoder 部分 ResBlock：
        # 1. Bottleneck 的 2 个 ResBlock -> FiLMResBlock
        # 2. Decoder 前 3 层 (level 0/1/2) 的 2xResBlock -> FiLMResBlock
        # 3. Encoder 和 Decoder 最浅层保持不变
    # Args:
        # base_ch:      基础通道数（默认 32，与 A方案一致）
        # in_channels:  输入通道（默认 3，RGB）
        # out_channels: 输出通道（默认 3，RGB）
        # embed_dim:    退化嵌入维度（默认 64）
    def __init__(self, base_ch=32, in_channels=3, out_channels=3, embed_dim=64):
        super().__init__(base_ch, in_channels, out_channels)

        self.embed_dim = embed_dim
        enc_chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        dec_chs = [enc_chs[-1], enc_chs[2], enc_chs[1], enc_chs[0]]  # [8x, 4x, 2x, 1x]

        # 退化估计器
        self.deg_estimator = DegEstimator(in_channels, embed_dim)

        # Bottleneck: 替换为 FiLMResBlock
        self.bottleneck = nn.ModuleList([
            FiLMResBlock(enc_chs[-1]),
            FiLMResBlock(enc_chs[-1]),
        ])
        self.film_bottleneck = nn.ModuleList([
            FiLM(embed_dim, enc_chs[-1]),
            FiLM(embed_dim, enc_chs[-1]),
        ])

        # Decoder: 前3层(level 0/1/2)替换为 FiLMResBlock
        # level 3（最浅层）保持原样
        self.film_decoder = nn.ModuleList()
        for i in range(4):
            out_ch = dec_chs[i]
            skip_ch = enc_chs[3 - i]

            if i < 3:  # level 0,1,2 插入 FiLM
                self.dec_blocks[i] = nn.ModuleList([
                    nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                    FiLMResBlock(out_ch),
                    FiLMResBlock(out_ch),
                ])
                self.film_decoder.append(nn.ModuleList([
                    FiLM(embed_dim, out_ch),
                    FiLM(embed_dim, out_ch),
                ]))
            else:  # level 3 保持标准 ResBlock
                self.dec_blocks[i] = nn.ModuleList([
                    nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                    ResBlock(out_ch),
                    ResBlock(out_ch),
                ])
                self.film_decoder.append(None)

    def forward(self, x):
        identity = x

        # 初始特征
        x = self.init_conv(x)

        # 编码器（不插 FiLM）
        skips = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            skips.append(x)
            x = self.down_blocks[i](x)

        # 退化嵌入（只算一次，供深层复用）
        deg_embed = self.deg_estimator(identity)

        # 瓶颈层（插入 FiLM）
        for idx, res in enumerate(self.bottleneck):
            film_params = self.film_bottleneck[idx](deg_embed)
            x = res(x, film_params)

        # 解码器（前3层插入 FiLM，最浅层不插）
        for i in range(4):
            x = self.up_blocks[i](x)
            skip = skips[3 - i]

            if i < 3:
                # 显式处理 FiLM 注入，不走 _fuse_skip
                conv = self.dec_blocks[i][0]
                x = conv(torch.cat([x, skip], dim=1))
                for j in range(2):
                    film_params = self.film_decoder[i][j](deg_embed)
                    x = self.dec_blocks[i][j + 1](x, film_params)
            else:
                # 最浅层：标准 _fuse_skip
                x = self._fuse_skip(x, skip, i)

        # 输出头
        x = self.out_conv(x)
        return x + identity


__all__ = ['DegFiLMResUNet']
