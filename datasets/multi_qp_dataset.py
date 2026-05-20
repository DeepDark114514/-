import random
import torch
from torch.utils.data import Dataset, DataLoader

from .mfqev2_dataset import MFQEv2Dataset


# 多QP训练用，每次随机选一个QP来加载，模拟多压缩强度
class MultiQPDataset(MFQEv2Dataset):
    def __init__(self, root, split, list_file, patch_size=256, mode='train', qp_list=None):
        if qp_list is None:
            qp_list = [22, 32, 42]
        super().__init__(root=root, split=split, list_file=list_file,
                         patch_size=patch_size, mode=mode, qp=qp_list)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.index_table[idx]
        meta = self.seq_meta[seq_idx]

        if self.mode == 'train' and len(self.qp_list) > 1:
            qp = random.choice(self.qp_list)  # 每个batch随机QP，让B方案学退化自适应
        else:
            qp = self.qp_list[0]

        lq, hq = self._load_frame(meta, frame_idx, qp=qp)

        if self.mode == 'train':
            lq, hq = self._crop_or_pad(lq, hq)
            lq, hq = self._augment(lq, hq)

        lq = torch.from_numpy(lq.transpose(2, 0, 1))
        hq = torch.from_numpy(hq.transpose(2, 0, 1))
        qp_tensor = torch.tensor(qp, dtype=torch.int32)

        if self.mode == 'train':
            return lq, hq, qp_tensor
        else:
            return lq, hq, meta['name']


def build_multi_qp_dataloader(cfg: dict, split: str, qp_list=None):
    is_train = (split == 'train')
    if qp_list is None:
        qp_list = cfg.get('qp_list', [22, 32, 42])

    dataset = MultiQPDataset(
        root=cfg['root'],
        split=split,
        list_file=cfg[f'{split}_list'],
        patch_size=cfg['patch_size'],
        mode='train' if is_train else 'eval',
        qp_list=qp_list if is_train else cfg.get('qp', 32),
    )

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


__all__ = ['MultiQPDataset', 'build_multi_qp_dataloader']
