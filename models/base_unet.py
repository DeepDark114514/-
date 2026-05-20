import torch
import torch.nn as nn


# Pre-activation：BN-ReLU先，再卷积。比Post-activation好训一点，但FP16会炸
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x, side_input=None):
        out = self.body(x)
        return x + out


class BaseUNet(nn.Module):
    def __init__(self, base_ch=64, in_channels=3, out_channels=3):
        super().__init__()
        self.base_ch = base_ch
        enc_chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(4):
            enc_in_ch = enc_chs[i]
            down_out_ch = enc_chs[i + 1] if i < 3 else enc_chs[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(enc_in_ch), ResBlock(enc_in_ch),
            ]))
            self.down_blocks.append(nn.Conv2d(enc_in_ch, down_out_ch, 3,
                                             stride=2, padding=1, bias=False))

        self.bottleneck = nn.ModuleList([
            ResBlock(enc_chs[-1]), ResBlock(enc_chs[-1]),
        ])

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_chs = [enc_chs[-1], enc_chs[2], enc_chs[1], enc_chs[0]]
        skip_chs = list(enc_chs)

        for i in range(4):
            in_ch = dec_chs[i - 1] if i > 0 else enc_chs[-1]
            out_ch = dec_chs[i]
            skip_ch = skip_chs[3 - i]
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            ))
            self.dec_blocks.append(nn.ModuleList([
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                ResBlock(out_ch), ResBlock(out_ch),
            ]))

        self.out_conv = nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=False)  # 输出层不加激活，回归像素值

    def _fuse_skip(self, dec_feat, skip_feat, level):
        x = torch.cat([dec_feat, skip_feat], dim=1)
        conv, res1, res2 = self.dec_blocks[level]
        x = conv(x)
        x = res1(x)
        x = res2(x)
        return x

    def forward(self, x):
        # 残差连接，末尾加回来
        identity = x
        x = self.init_conv(x)

        # encoder: 4个stage，每个stage两个ResBlock
        skips = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            skips.append(x)
            x = self.down_blocks[i](x)

        for res in self.bottleneck:
            x = res(x)

        for i in range(4):
            x = self.up_blocks[i](x)
            skip = skips[3 - i]
            x = self._fuse_skip(x, skip, i)

        x = self.out_conv(x)
        return x + identity  # 全局残差：学的是残差而不是直接生成，收敛快

    def get_encoder_features(self, x):
        x = self.init_conv(x)
        feats = []
        for i in range(4):
            for res in self.enc_blocks[i]:
                x = res(x)
            feats.append(x)
            x = self.down_blocks[i](x)
        return feats, x
