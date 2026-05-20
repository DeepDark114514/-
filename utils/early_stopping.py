#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# 早停机制：验证集指标连续 N 个周期不提升，自动停止并保存最佳模型

import os
import torch


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode='max'):
        # Args:
            # patience: 连续多少个 epoch 无提升则停止
            # min_delta: 提升阈值，小于此值视为无提升
            # mode: 'max' 表示指标越大越好（如 PSNR），'min' 表示越小越好（如 loss）
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_best_model(self, model, optimizer, epoch, path='checkpoints/best_model.pth'):
        # 保存当前最佳模型（应在验证后调用）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': self.best_score,
        }, path)

    def load_best_model(self, model, optimizer=None, path='checkpoints/best_model.pth'):
        # 加载最佳模型
        if not os.path.exists(path):
            return None
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
