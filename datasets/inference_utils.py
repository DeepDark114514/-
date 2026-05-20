import numpy as np
import torch
import torch.nn.functional as F


# 大图分块预测，防止显存爆炸，重叠区域取平均
# stride=tile_size//2，重叠50%然后加权平均，消除边界
def tile_predict(model, lq_frame, tile_size=256, stride=128):
    if lq_frame.dim() == 3:
        lq_frame = lq_frame.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, c, h, w = lq_frame.shape

    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            out = model(lq_frame)
        return out.squeeze(0) if squeeze else out

    out = torch.zeros((1, c, h, w), dtype=lq_frame.dtype, device=lq_frame.device)
    weight = torch.zeros((1, 1, h, w), dtype=lq_frame.dtype, device=lq_frame.device)

    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = lq_frame[:, :, y_start:y_end, x_start:x_end]

                pad_h = tile_size - (y_end - y_start)
                pad_w = tile_size - (x_end - x_start)
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

                pred_tile = model(tile)

                if pad_h > 0 or pad_w > 0:
                    pred_tile = pred_tile[:, :, :tile_size - pad_h, :tile_size - pad_w]

                out[:, :, y_start:y_end, x_start:x_end] += pred_tile
                weight[:, :, y_start:y_end, x_start:x_end] += 1.0

    out = out / weight.clamp(min=1.0)  # 重叠区域被加了多次，除一下做平均
    return out.squeeze(0) if squeeze else out


def pad_frame(frame, target_size=256):
    _, c, h, w = frame.shape
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_h > 0 or pad_w > 0:
        frame = F.pad(frame, (0, pad_w, 0, pad_h), mode='reflect')
    return frame, (pad_h, pad_w)
