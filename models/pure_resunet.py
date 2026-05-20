from .base_unet import BaseUNet, ResBlock


class PureResUNet(BaseUNet):
    def __init__(self, base_ch=32, in_channels=3, out_channels=3):
        super().__init__(base_ch, in_channels, out_channels)


__all__ = ['PureResUNet', 'ResBlock']
