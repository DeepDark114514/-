#  南京信息工程大学22级信安1班 202283290014
# 2026.5.11
# Tile-based 推理工具（用于验证/测试阶段）
# 对于高分辨率帧（如 1920x1080），直接整帧输入可能导致显存溢出。
# 本模块提供 tile-based 推理：将大图切成 256x256 的 patch，
# 步长 128（重叠 128px），逐 patch 通过网络后重叠区域平均融合。

import numpy as np
import torch
import torch.nn.functional as F


def tile_predict(model, lq_frame, tile_size=256, stride=128):
    # 对单帧进行 tile-based 推理。
    # Args:
        # model: 神经网络模型，输入 (B, 3, H, W) 输出 (B, 3, H, W)
        # lq_frame: torch.Tensor, shape (1, 3, H, W) 或 (3, H, W)
        # tile_size: patch 大小
        # stride: 步长（overlap = tile_size - stride）
    # Returns:
        # torch.Tensor, shape (1, 3, H, W) 或 (3, H, W)
    if lq_frame.dim() == 3:
        lq_frame = lq_frame.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, c, h, w = lq_frame.shape

    # 如果帧本身就很小，直接推理
    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            out = model(lq_frame)
        return out.squeeze(0) if squeeze else out

    # 创建输出 buffer 和权重 buffer（用于平均融合）
    out = torch.zeros((1, c, h, w), dtype=lq_frame.dtype, device=lq_frame.device)
    weight = torch.zeros((1, 1, h, w), dtype=lq_frame.dtype, device=lq_frame.device)

    # 滑窗裁剪
    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = lq_frame[:, :, y_start:y_end, x_start:x_end]

                # 如果 tile 不是完整大小，pad 到 tile_size
                pad_h = tile_size - (y_end - y_start)
                pad_w = tile_size - (x_end - x_start)
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

                pred_tile = model(tile)

                # 去除 pad 部分
                if pad_h > 0 or pad_w > 0:
                    pred_tile = pred_tile[:, :, :tile_size - pad_h, :tile_size - pad_w]

                out[:, :, y_start:y_end, x_start:x_end] += pred_tile
                weight[:, :, y_start:y_end, x_start:x_end] += 1.0

    # 平均融合
    out = out / weight.clamp(min=1.0)

    return out.squeeze(0) if squeeze else out


def pad_frame(frame, target_size=256):
    # 将小于 target_size 的帧 pad 到 target_size（reflect padding）
    # 用于 Class D (416x240) 等低分辨率序列的直接推理。
    _, c, h, w = frame.shape
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_h > 0 or pad_w > 0:
        frame = F.pad(frame, (0, pad_w, 0, pad_h), mode='reflect')
    return frame, (pad_h, pad_w)
