import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .yuv_io import read_yuv


# MFQEv2 数据集，从 YUV 文件逐帧读取并返回 LQ/HQ 对
class MFQEv2Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        list_file: str,
        patch_size: int = 256,  # 256x256 patch，显存和速度的平衡点
        mode: str = 'train',
        qp=32,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.mode = mode
        self.qp_list = [qp] if isinstance(qp, int) else list(qp)

        list_path = self.root / list_file
        with open(list_path, 'r', encoding='utf-8') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        self.seq_meta = []
        for seq_name in self.sequences:
            gt_path = self.root / 'gt' / split / f"{seq_name}.yuv"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT file for {seq_name}")

            lq_paths = {}
            for qp_val in self.qp_list:
                lq_path = self.root / 'compressed' / split / f"{seq_name}_qp{qp_val}.yuv"
                if not lq_path.exists():
                    raise FileNotFoundError(f"Missing LQ file for {seq_name} QP{qp_val}")
                lq_paths[qp_val] = str(lq_path)

            import re
            match = re.search(r'(\d+)x(\d+)', seq_name)
            if not match:
                raise ValueError(f"Cannot parse resolution from {seq_name}")
            w, h = int(match.group(1)), int(match.group(2))

            frame_size = w * h * 3 // 2
            file_size = gt_path.stat().st_size
            num_frames = file_size // frame_size

            self.seq_meta.append({
                'name': seq_name,
                'gt_path': str(gt_path),
                'lq_paths': lq_paths,
                'width': w,
                'height': h,
                'num_frames': num_frames,
            })

        self.index_table = []
        for seq_idx, meta in enumerate(self.seq_meta):
            for frame_idx in range(meta['num_frames']):
                self.index_table.append((seq_idx, frame_idx))

        print(f"[MFQEv2Dataset] {split}: {len(self.sequences)} seqs, "
              f"{'train' if mode == 'train' else 'eval'} mode, "
              f"{len(self.index_table)} samples")

    def __len__(self):
        return len(self.index_table)

    def _load_frame(self, meta: dict, frame_idx: int, qp: int = None):
        w, h = meta['width'], meta['height']
        frame_size_y = w * h
        frame_size_uv = (w // 2) * (h // 2)
        frame_bytes = frame_size_y + 2 * frame_size_uv
        file_offset = frame_idx * frame_bytes

        def read_single_frame(path: str, offset: int):
            with open(path, 'rb') as f:
                f.seek(offset)
                raw = f.read(frame_bytes)
            yuv_img = np.frombuffer(raw, dtype=np.uint8).reshape((h + h // 2, w))
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
            return rgb.astype(np.float32) / 255.0

        lq_path = meta['lq_paths'][qp] if qp and qp in meta['lq_paths'] else meta['lq_paths'][self.qp_list[0]]
        lq = read_single_frame(lq_path, file_offset)
        hq = read_single_frame(meta['gt_path'], file_offset)
        return lq, hq

    def _crop_or_pad(self, lq: np.ndarray, hq: np.ndarray):
        h, w = lq.shape[:2]
        ps = self.patch_size

        if h < ps or w < ps:
            # 小帧reflect pad到patch_size，不然没法裁
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            lq = np.pad(lq, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            hq = np.pad(hq, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = lq.shape[:2]

        if h == ps and w == ps:
            top, left = 0, 0
        else:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)

        lq = lq[top:top+ps, left:left+ps, :]
        hq = hq[top:top+ps, left:left+ps, :]
        return lq, hq

    def _augment(self, lq: np.ndarray, hq: np.ndarray):
        if random.random() < 0.5:
            lq = np.fliplr(lq).copy()
            hq = np.fliplr(hq).copy()
        return lq, hq

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.index_table[idx]
        meta = self.seq_meta[seq_idx]

        if self.mode == 'train' and len(self.qp_list) > 1:
            qp = random.choice(self.qp_list)
        else:
            qp = self.qp_list[0]

        lq, hq = self._load_frame(meta, frame_idx, qp=qp)

        if self.mode == 'train':
            lq, hq = self._crop_or_pad(lq, hq)
            lq, hq = self._augment(lq, hq)

        lq = torch.from_numpy(lq.transpose(2, 0, 1))
        hq = torch.from_numpy(hq.transpose(2, 0, 1))

        if self.mode == 'train':
            return lq, hq
        else:
            return lq, hq, meta['name']


def build_dataloader(cfg: dict, split: str, qp: int = None):
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

    # eval别开多进程，Windows下num_workers>0容易锁死
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
