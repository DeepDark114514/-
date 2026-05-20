import torch
import torch.nn as nn

from .base_unet import BaseUNet, ResBlock
from .deg_film_blocks import DegEstimator, FiLM, FiLMResBlock


# B方案: 带退化感知的U-Net，U-Net部分复用BaseUNet
class DegFiLMResUNet(BaseUNet):
    def __init__(self, base_ch=32, in_channels=3, out_channels=3, embed_dim=64):
        super().__init__(base_ch, in_channels, out_channels)

        self.embed_dim = embed_dim
        enc_chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        dec_chs = [enc_chs[-1], enc_chs[2], enc_chs[1], enc_chs[0]]

        self.deg_estimator = DegEstimator(in_channels, embed_dim)

        self.bottleneck = nn.ModuleList([
            FiLMResBlock(enc_chs[-1]),
            FiLMResBlock(enc_chs[-1]),
        ])
        self.film_bottleneck = nn.ModuleList([
            FiLM(embed_dim, enc_chs[-1]),
            FiLM(embed_dim, enc_chs[-1]),
        ])

        self.film_decoder = nn.ModuleList()
        for i in range(4):
            out_ch = dec_chs[i]
            skip_ch = enc_chs[3 - i]

            if i < 3:  # 最浅层(decoder第4层)不加FiLM，特征太细了估计退化没意义
                self.dec_blocks[i] = nn.ModuleList([
                    nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                    FiLMResBlock(out_ch),
                    FiLMResBlock(out_ch),
                ])
                self.film_decoder.append(nn.ModuleList([
                    FiLM(embed_dim, out_ch),
                    FiLM(embed_dim, out_ch),
                ]))
            else:
                self.dec_blocks[i] = nn.ModuleList([
                    nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                    ResBlock(out_ch),
                    ResBlock(out_ch),
                ])
                self.film_decoder.append(None)

    def forward(self, x):
        identity = x
        x = self.init_conv(x)

        skips = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            skips.append(x)
            x = self.down_blocks[i](x)

        deg_embed = self.deg_estimator(identity)  # 用原始LQ帧估计退化，不是用特征

        for idx, res in enumerate(self.bottleneck):
            film_params = self.film_bottleneck[idx](deg_embed)
            x = res(x, film_params)

        for i in range(4):
            x = self.up_blocks[i](x)
            skip = skips[3 - i]

            if i < 3:
                conv = self.dec_blocks[i][0]
                x = conv(torch.cat([x, skip], dim=1))
                for j in range(2):
                    film_params = self.film_decoder[i][j](deg_embed)
                    x = self.dec_blocks[i][j + 1](x, film_params)
            else:
                x = self._fuse_skip(x, skip, i)

        x = self.out_conv(x)
        return x + identity


__all__ = ['DegFiLMResUNet']
