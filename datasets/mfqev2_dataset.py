#  南京信息工程大学22级信安1班 202283290014
# 2026.5.11
# MFQEv2 PyTorch Dataset & DataLoader（A 方案基线）
# 训练模式：
  # - 从 compressed/train/ 和 gt/train/ 读取
  # - 单 QP：只读 _qp32.yuv
  # - 随机裁剪 256×256 patch（LQ 和 HQ 同一坐标）
  # - 如果帧高或帧宽 < 256，使用整帧并 reflect-pad 到 256
  # - 数据增强：仅随机水平翻转
  # - 返回 (lq_patch, hq_patch), shape (3, 256, 256)
# 验证/测试模式：
  # - 从 compressed/val/（或 test）和 gt/val/（或 test）读取
  # - 整帧输入，不做随机裁剪
  # - 返回整帧 (lq_frame, hq_frame), shape (3, H, W)
  # - Tile-based 推理由外部 inference 函数处理

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .yuv_io import read_yuv


class MFQEv2Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,           # 'train', 'val', 'test'
        list_file: str,
        patch_size: int = 256,
        mode: str = 'train',  # 'train' or 'eval'
        qp=32,                # 支持 int 单QP 或 list 多QP混合
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.mode = mode
        # 统一为列表：单QP -> [qp], 多QP -> qp
        self.qp_list = [qp] if isinstance(qp, int) else list(qp)

        # 读取序列列表
        list_path = self.root / list_file
        with open(list_path, 'r', encoding='utf-8') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        # 为每个序列预加载元数据（所有QP共享分辨率、帧数，LQ路径不同）
        self.seq_meta = []
        for seq_name in self.sequences:
            gt_path = self.root / 'gt' / split / f"{seq_name}.yuv"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT file for {seq_name}")

            # 收集所有可用QP的LQ路径
            lq_paths = {}
            for qp_val in self.qp_list:
                lq_path = self.root / 'compressed' / split / f"{seq_name}_qp{qp_val}.yuv"
                if not lq_path.exists():
                    raise FileNotFoundError(f"Missing LQ file for {seq_name} QP{qp_val}")
                lq_paths[qp_val] = str(lq_path)

            # 从文件名解析分辨率
            import re
            match = re.search(r'(\d+)x(\d+)', seq_name)
            if not match:
                raise ValueError(f"Cannot parse resolution from {seq_name}")
            w, h = int(match.group(1)), int(match.group(2))

            # 计算帧数
            frame_size = w * h * 3 // 2
            file_size = gt_path.stat().st_size
            num_frames = file_size // frame_size

            self.seq_meta.append({
                'name': seq_name,
                'gt_path': str(gt_path),
                'lq_paths': lq_paths,   # dict: qp -> path
                'width': w,
                'height': h,
                'num_frames': num_frames,
            })

        # 训练模式：构建 (seq_idx, frame_idx) 索引表
        if self.mode == 'train':
            self.index_table = []
            for seq_idx, meta in enumerate(self.seq_meta):
                for frame_idx in range(meta['num_frames']):
                    self.index_table.append((seq_idx, frame_idx))
        else:
            # eval 模式：构建 (seq_idx, frame_idx) 索引表，逐帧返回避免内存爆炸
            self.index_table = []
            for seq_idx, meta in enumerate(self.seq_meta):
                for frame_idx in range(meta['num_frames']):
                    self.index_table.append((seq_idx, frame_idx))

        print(f"[MFQEv2Dataset] {split}: {len(self.sequences)} sequences, "
              f"{'train' if mode == 'train' else 'eval'} mode, "
              f"{len(self.index_table)} samples")

    def __len__(self):
        return len(self.index_table)

    def _load_frame(self, meta: dict, frame_idx: int, qp: int = None) -> tuple:
        # 加载指定帧的 LQ 和 HQ，返回 (lq_rgb, hq_rgb), shape (H, W, 3)
        # 使用 OpenCV 加速 YUV420p -> RGB 转换（比 numpy 实现快 ~4x）
        # qp: 指定QP，None则使用 self.qp_list[0]（单QP模式兼容）
        w, h = meta['width'], meta['height']
        frame_size_y = w * h
        frame_size_uv = (w // 2) * (h // 2)
        frame_bytes = frame_size_y + 2 * frame_size_uv
        file_offset = frame_idx * frame_bytes

        def read_single_frame(path: str, offset: int):
            with open(path, 'rb') as f:
                f.seek(offset)
                raw = f.read(frame_bytes)
            # I420 (YUV420p) -> RGB: Y 平面 + U 平面 + V 平面
            yuv_img = np.frombuffer(raw, dtype=np.uint8).reshape((h + h // 2, w))
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
            return rgb.astype(np.float32) / 255.0

        lq_path = meta['lq_paths'][qp] if qp and qp in meta['lq_paths'] else meta['lq_paths'][self.qp_list[0]]
        lq = read_single_frame(lq_path, file_offset)
        hq = read_single_frame(meta['gt_path'], file_offset)
        return lq, hq

    def _crop_or_pad(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        # 训练模式：随机裁剪 256x256；如果帧太小则 pad
        h, w = lq.shape[:2]
        ps = self.patch_size

        if h < ps or w < ps:
            # Reflect pad 到至少 patch_size
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            lq = np.pad(lq, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            hq = np.pad(hq, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = lq.shape[:2]

        # 随机裁剪同一位置
        if h == ps and w == ps:
            top, left = 0, 0
        else:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)

        lq = lq[top:top+ps, left:left+ps, :]
        hq = hq[top:top+ps, left:left+ps, :]
        return lq, hq

    def _augment(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        # 数据增强：仅随机水平翻转
        if random.random() < 0.5:
            lq = np.fliplr(lq).copy()
            hq = np.fliplr(hq).copy()
        return lq, hq

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.index_table[idx]
        meta = self.seq_meta[seq_idx]

        # 多QP混合训练：每个样本随机抽取一个QP
        if self.mode == 'train' and len(self.qp_list) > 1:
            qp = random.choice(self.qp_list)
        else:
            qp = self.qp_list[0]

        lq, hq = self._load_frame(meta, frame_idx, qp=qp)

        if self.mode == 'train':
            lq, hq = self._crop_or_pad(lq, hq)
            lq, hq = self._augment(lq, hq)

        # HWC -> CHW
        lq = torch.from_numpy(lq.transpose(2, 0, 1))
        hq = torch.from_numpy(hq.transpose(2, 0, 1))

        if self.mode == 'train':
            return lq, hq
        else:
            return lq, hq, meta['name']


def build_dataloader(cfg: dict, split: str, qp: int = None):
    # 根据配置构建 DataLoader
    # Args:
        # cfg: 配置字典
        # split: 'train', 'val', 'test'
        # qp: 指定 QP 值，默认使用 cfg['qp']
    is_train = (split == 'train')
    if qp is None:
        qp = cfg.get('qp', 32)
    dataset = MFQEv2Dataset(
        root=cfg['root'],
        split=split,
        list_file=cfg[f'{split}_list'],
        patch_size=cfg['patch_size'],
        mode='train' if is_train else 'eval',
        qp=qp,
    )

    # eval 模式 num_workers 直接设 0，避免 Windows 多进程死锁 + 内存爆炸
    nw = cfg['num_workers'] if is_train else 0
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'] if is_train else 1,
        shuffle=is_train,
        num_workers=nw,
        pin_memory=cfg['pin_memory'],
        drop_last=is_train,
        persistent_workers=cfg.get('persistent_workers', False) and (nw > 0),
    )
    return loader
