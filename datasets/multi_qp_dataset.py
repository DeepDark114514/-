#  南京信息工程大学22级信安1班 202283290014
# 2026.5.17
# 多QP分层采样数据集（B方案训练专用）
# 在 MFQEv2Dataset 基础上扩展，训练时额外返回当前样本的 QP 值，
# 便于训练脚本记录各 QP 的 loss 分布，验证盲QP自适应效果。
# 分层采样策略:
    # - 每个 (seq, frame) 组合在每次 __getitem__ 时，
      # 从 qp_list 中均匀随机抽取一个 QP
    # - 由于 DataLoader shuffle=True 会对 index_table 全局打乱，
      # 当 batch_size >= len(qp_list) 时，每个 batch 内各 QP 近似均匀分布
    # - 总样本数与单 QP 训练相同，不增加磁盘占用
# 与 A方案 MFQEv2Dataset 的区别:
    # - 训练模式返回 (lq, hq, qp_int) 而非 (lq, hq)
    # - qp_int 为标量张量，供日志或分析使用，模型前向不直接消费

import random
import torch
from torch.utils.data import Dataset, DataLoader

from .mfqev2_dataset import MFQEv2Dataset


class MultiQPDataset(MFQEv2Dataset):
    # 多QP分层采样数据集
    # Args:
        # root, split, list_file, patch_size, mode: 同 MFQEv2Dataset
        # qp_list: QP 列表，默认 [22, 32, 42]（与你现有数据一致）
                 # 如需扩展5档，改为 [22, 27, 32, 37, 42] 即可
    def __init__(self, root, split, list_file, patch_size=256, mode='train', qp_list=None):
        # 默认使用你现有的3个QP档位
        if qp_list is None:
            qp_list = [22, 32, 42]
        super().__init__(root=root, split=split, list_file=list_file,
                         patch_size=patch_size, mode=mode, qp=qp_list)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.index_table[idx]
        meta = self.seq_meta[seq_idx]

        # 训练模式：随机均匀采样一个 QP
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
        qp_tensor = torch.tensor(qp, dtype=torch.int32)

        if self.mode == 'train':
            return lq, hq, qp_tensor
        else:
            return lq, hq, meta['name']


def build_multi_qp_dataloader(cfg: dict, split: str, qp_list=None):
    # 构建多QP DataLoader（B方案专用）
    # Args:
        # cfg: 配置字典
        # split: 'train', 'val', 'test'
        # qp_list: 训练时混合的QP列表，默认使用 cfg.get('qp_list', [22,32,42])
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
