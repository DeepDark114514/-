#  南京信息工程大学22级信安1班 202283290014
# 2026.5.12  (refactored 2026.5.15)
# A方案：PureResUNet
# 纯像素驱动，零外部知识，零注意力，零条件化。
# 继承 BaseUNet 骨架，使用标准 skip connection（直接 concat + conv）。
# 所有网络结构定义在 BaseUNet 中统一维护，此处仅做方案标记与导出。
# State-dict 兼容性说明:
    # BaseUNet 的模块命名与原版 PureResUNet 完全一致，
    # 旧 checkpoint (logs/A_20260515_002558/best_model.pth) 可直接加载。

from .base_unet import BaseUNet, ResBlock


class PureResUNet(BaseUNet):
    # A方案：纯残差 U-Net
    # 标准 skip connection，不做任何频域增强或条件化。
    # 作为 baseline，用于与 B 方案 (DegFiLMResUNet) 做消融对比。
    pass


# 保持向后兼容的别名导出
__all__ = ['PureResUNet', 'ResBlock']
