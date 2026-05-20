import numpy as np
from pathlib import Path


def read_yuv(path: str, width: int, height: int):
    frame_size_y = width * height
    frame_size_uv = (width // 2) * (height // 2)
    frame_size_total = frame_size_y + 2 * frame_size_uv

    file_size = Path(path).stat().st_size
    num_frames = file_size // frame_size_total

    y = np.zeros((num_frames, height, width), dtype=np.uint8)
    u = np.zeros((num_frames, height // 2, width // 2), dtype=np.uint8)
    v = np.zeros((num_frames, height // 2, width // 2), dtype=np.uint8)

    with open(path, "rb") as f:
        for i in range(num_frames):
            y[i] = np.frombuffer(f.read(frame_size_y), dtype=np.uint8).reshape((height, width))
            u[i] = np.frombuffer(f.read(frame_size_uv), dtype=np.uint8).reshape((height // 2, width // 2))
            v[i] = np.frombuffer(f.read(frame_size_uv), dtype=np.uint8).reshape((height // 2, width // 2))

    u_up = np.repeat(np.repeat(u, 2, axis=1), 2, axis=2)
    v_up = np.repeat(np.repeat(v, 2, axis=1), 2, axis=2)

    y_f = y.astype(np.float32)
    u_f = u_up.astype(np.float32) - 128.0
    v_f = v_up.astype(np.float32) - 128.0

    r = y_f + 1.402 * v_f
    g = y_f - 0.344136 * u_f - 0.714136 * v_f
    b = y_f + 1.772 * u_f

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0)
    rgb = rgb / 255.0

    return rgb.astype(np.float32)


def read_yuv_y_only(path: str, width: int, height: int):
    import numpy as np
    from pathlib import Path

    frame_size_y = width * height
    frame_size_uv = (width // 2) * (height // 2)
    frame_size_total = frame_size_y + 2 * frame_size_uv

    file_size = Path(path).stat().st_size
    num_frames = file_size // frame_size_total

    y = np.zeros((num_frames, height, width), dtype=np.uint8)
    with open(path, "rb") as f:
        for i in range(num_frames):
            y[i] = np.frombuffer(f.read(frame_size_y), dtype=np.uint8).reshape((height, width))
            f.read(2 * frame_size_uv)

    return (y.astype(np.float32) / 255.0)
