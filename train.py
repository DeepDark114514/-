#  南京信息工程大学22级信安1班 202283290014
# 2026.5.10
# A/B/C 统一训练入口
# 一键切换网络、损失、优化器、数据流完全不变
# 命令行参数覆盖 config.py 示例:
  # python train.py --model_type A --base_ch 32 --epochs 100
  # python train.py -m B --lr 1e-4
  # python train.py -m C --base_ch 32 --batch_size 8

import os
import sys
import time
import random
import argparse
import logging
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import CONFIG
from models import PureResUNet, DegFiLMResUNet
from losses import L1SSIMLoss
from data import build_dataloader, tile_predict
from utils import EarlyStopping, calc_psnr, set_high_priority, disable_quick_edit_tip


def parse_args():
    # 命令行参数解析，允许在不修改 config.py 的情况下切换 A/B/C
    parser = argparse.ArgumentParser(description='A 方案训练框架')
    parser.add_argument('-m', '--model_type', type=str, default=None,
                        choices=['A', 'B'],
                        help="网络方案: A=PureResUNet(基线), B=DegFiLMResUNet(退化感知FiLM)")
    parser.add_argument('--base_ch', type=int, default=None,
                        help="基础通道数 (默认 32)")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="训练 batch size (默认 8)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="训练 epoch 数 (默认 100)")
    parser.add_argument('--lr', type=float, default=None,
                        help="学习率 (默认 1e-4)")
    parser.add_argument('--model', type=str, default=None,
                        help="模型方案简写: A/B/C (同 --model_type)")
    parser.add_argument('--resume', type=str, default=None,
                        help="从指定 checkpoint 恢复训练 (如: checkpoints/epoch_20.pth)")
    return parser.parse_args()


def override_config(cfg, args):
    # 用命令行参数覆盖 config
    # 支持 --model 或 --model_type（当前仅支持 A）
    model_type = args.model if args.model else args.model_type
    if model_type is not None:
        if model_type not in ('A', 'B'):
            raise ValueError("当前仅支持 A/B 方案")
        cfg['model_type'] = model_type
    if args.base_ch is not None:
        cfg['base_ch'] = args.base_ch
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.lr is not None:
        cfg['lr'] = args.lr
    return cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    # 根据 model_type 构建对应网络
    model_type = cfg.get('model_type', 'A')
    if model_type == 'A':
        return PureResUNet(base_ch=cfg['base_ch'])
    elif model_type == 'B':
        return DegFiLMResUNet(base_ch=cfg['base_ch'])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_optimizer_scheduler(model, cfg):
    # 构建优化器和学习率调度器（A/B/C 完全一致）
    optimizer_name = cfg['optimizer']
    lr = cfg['lr']
    wd = cfg['weight_decay']

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler_name = cfg['scheduler']
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg['step_size'], gamma=cfg['gamma']
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs']
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, cfg, epoch, logger):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    grad_norm = 0.0
    pbar = tqdm(dataloader, desc='Train', leave=False)

    accum_steps = cfg.get('grad_accum_steps', 1)
    optimizer.zero_grad()

    for step, (lq, hq) in enumerate(pbar):
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        if cfg['amp']:
            with autocast('cuda'):
                pred = model(lq)
                if cfg.get('pred_clamp', False):
                    pred = torch.clamp(pred, 0.0, 1.0)
                loss, l1_val, ssim_loss_val, ssim_val = loss_fn(pred, hq, return_components=True)
                loss = loss / accum_steps
        else:
            pred = model(lq)
            if cfg.get('pred_clamp', False):
                pred = torch.clamp(pred, 0.0, 1.0)
            loss, l1_val, ssim_loss_val, ssim_val = loss_fn(pred, hq, return_components=True)
            loss = loss / accum_steps

        # 记录监控指标（no_grad 避免影响梯度）
        with torch.no_grad():
            psnr_val = calc_psnr(pred.clamp(0, 1), hq)

        if cfg['amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每 accum_steps 步执行一次优化器更新
        if (step + 1) % accum_steps == 0:
            if cfg['amp']:
                if cfg.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.get('clip_grad_norm', None):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_l1 += l1_val
        total_ssim += ssim_val
        total_psnr += psnr_val
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})

        # 每 50 个 batch 记录日志
        if step % 50 == 0:
            logger.info(f"Epoch {epoch}/{cfg['epochs']}, Batch {step+1}/{len(dataloader)}, Loss: {loss.item() * accum_steps:.4f}, L1: {l1_val:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}")

    # 处理最后不足 accum_steps 的残余梯度
    if (step + 1) % accum_steps != 0:
        if cfg['amp']:
            if cfg.get('clip_grad_norm', None):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.get('clip_grad_norm', None):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
        optimizer.zero_grad()

    n = len(dataloader)
    return total_loss / n, float(grad_norm), total_l1 / n, total_ssim / n, total_psnr / n


def _pad_to_multiple(x, multiple=16):
    # 对称 pad 到 multiple 的倍数，返回 pad 后的张量和 (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='reflect')
    return x, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)


@torch.no_grad()
def validate(model, dataloader, device, cfg, loss_fn):
    # 验证：逐帧处理，同时计算 PSNR / SSIM / Loss
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_frames = 0

    for lq, hq, _ in tqdm(dataloader, desc='Val', leave=False):
        lq = lq.to(device)
        hq = hq.to(device)
        _, _, h, w = lq.shape

        use_tile = (h > 720 or w > 1280)

        if use_tile:
            pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
        else:
            lq_padded, pads = _pad_to_multiple(lq, multiple=16)
            try:
                pred = model(lq_padded)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
                    loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
                    total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
                    total_ssim += ssim_val
                    total_loss += loss.item()
                    total_frames += 1
                    continue
                else:
                    raise
            pht, phb, pwl, pwr = pads
            pred = pred[:, :, pht:pht + h, pwl:pwl + w]

        loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
        total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
        total_ssim += ssim_val
        total_loss += loss.item()
        total_frames += 1

    if total_frames == 0:
        return 0.0, 0.0, 0.0
    return total_psnr / total_frames, total_ssim / total_frames, total_loss / total_frames


@torch.no_grad()
def test(model, dataloader, device, cfg, loss_fn):
    # 测试：同验证逻辑，同步计算 PSNR / SSIM / Loss
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_frames = 0

    for lq, hq, _ in tqdm(dataloader, desc='Test', leave=False):
        lq = lq.to(device)
        hq = hq.to(device)
        _, _, h, w = lq.shape

        use_tile = (h > 720 or w > 1280)

        if use_tile:
            pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
        else:
            lq_padded, pads = _pad_to_multiple(lq, multiple=16)
            try:
                pred = model(lq_padded)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
                    loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
                    total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
                    total_ssim += ssim_val
                    total_loss += loss.item()
                    total_frames += 1
                    continue
                else:
                    raise
            pht, phb, pwl, pwr = pads
            pred = pred[:, :, pht:pht + h, pwl:pwl + w]

        loss, _, _, ssim_val = loss_fn(pred.clamp(0, 1), hq, return_components=True)
        total_psnr += float(calc_psnr(pred.clamp(0, 1), hq))
        total_ssim += ssim_val
        total_loss += loss.item()
        total_frames += 1

    if total_frames == 0:
        return 0.0, 0.0, 0.0
    return total_psnr / total_frames, total_ssim / total_frames, total_loss / total_frames


def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logger(exp_dir):
    # 配置日志：同时输出到控制台和文件，日志写入实验目录
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(exp_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（如 notebook 中多次调用）
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_args()
    cfg = CONFIG.copy()
    cfg = override_config(cfg, args)
    set_seed(cfg['seed'])
    set_high_priority()

    # 实验目录（隔离不同运行的产物）
    if args.resume:
        # 从 checkpoint 路径推断实验目录
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
        if not os.path.exists(exp_dir):
            raise ValueError(f"无法从 checkpoint 路径推断实验目录: {args.resume}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg['model_type']}_{timestamp}"
        exp_dir = os.path.join('logs', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)

    # 保存配置到实验目录
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 日志系统
    logger = setup_logger(exp_dir)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"开始训练方案: {cfg['model_type']}")
    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"配置: {cfg}")
    logger.info(f"设备: {device}")

    # cuDNN 自动寻找最优卷积算法（固定输入尺寸时显著加速）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark: enabled")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. 一键切换网络
    model = build_model(cfg)
    model = model.to(device)
    n_params = count_parameters(model)
    logger.info(f"模型参数量: {n_params:,} ({n_params / 1e6:.2f}M)")

    # torch.compile: PyTorch 2.x 图编译加速（RTX 5080 + cu128 专属红利）
    # 注意：Windows 上需 triton 后端支持，若不可用则自动回退到 eager mode
    if hasattr(torch, 'compile'):
        try:
            import importlib.util
            triton_available = importlib.util.find_spec("triton") is not None
            if triton_available:
                model = torch.compile(model, mode="max-autotune")
                logger.info("torch.compile: enabled (mode=max-autotune)")
            else:
                logger.info("torch.compile: skipped (triton not available on Windows)")
        except Exception as e:
            logger.info(f"torch.compile failed: {e}, falling back to eager mode")

    # 2. 损失函数（A/B/C 完全一致）
    loss_fn = L1SSIMLoss(
        l1_weight=cfg['l1_weight'],
        ssim_weight=cfg['ssim_weight']
    )
    loss_fn = loss_fn.to(device)

    # 3. 优化器 + 调度器
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    logger.info(f"优化器: {cfg['optimizer']}, lr={cfg['lr']}, scheduler={cfg.get('scheduler', 'None')}")

    # 4. 数据流
    # 训练：多QP混合（每个batch随机抽取不同QP）
    # 验证：固定QP32，保证指标可比性
    train_qp = cfg.get('qp_list', cfg.get('qp', 32))
    val_qp = cfg.get('qp', 32)
    train_loader = build_dataloader(cfg, 'train', qp=train_qp)
    val_loader = build_dataloader(cfg, 'val', qp=val_qp)
    logger.info(f"Train batches: {len(train_loader)}, Val sequences: {len(val_loader)}")

    # 5. 早停
    early_stopper = EarlyStopping(
        patience=cfg['early_stop_patience'],
        min_delta=cfg['early_stop_min_delta'],
        mode=cfg['early_stop_mode']
    )
    logger.info(f"早停 patience={cfg['early_stop_patience']}, min_delta={cfg['early_stop_min_delta']}, mode={cfg['early_stop_mode']}")

    # 6. 恢复训练
    start_epoch = 1
    best_psnr = -1.0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', -1.0)
        logger.info(f"从 checkpoint 恢复: {args.resume}")
        logger.info(f"恢复 epoch {start_epoch - 1}, best_psnr: {best_psnr:.4f}")

    # 7. 训练循环
    scaler = GradScaler('cuda') if cfg['amp'] else None

    # CSV 指标记录（用于后续画曲线）
    import csv
    csv_path = os.path.join(exp_dir, 'training_metrics.csv')
    if args.resume and os.path.exists(csv_path):
        logger.info(f"继续写入已有 CSV: {csv_path}")
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_l1', 'train_ssim', 'train_psnr', 'grad_norm', 'val_psnr', 'val_ssim', 'val_loss', 'test_psnr', 'test_ssim', 'test_loss', 'lr'])

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        epoch_start = time.time()
        val_psnr = None
        val_ssim = None
        val_loss = None
        test_psnr = None
        test_ssim = None
        test_loss = None

        train_loss, grad_norm, train_l1, train_ssim, train_psnr = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, cfg, epoch, logger
        )
        logger.info(f"Epoch {epoch}/{cfg['epochs']} training summary - Loss: {train_loss:.6f}, L1: {train_l1:.6f}, SSIM: {train_ssim:.4f}, PSNR: {train_psnr:.2f}, Grad Norm: {grad_norm:.4f}")

        # 验证
        val_psnr = None
        val_ssim = None
        val_loss = None
        if epoch % cfg['val_interval'] == 0 or epoch == cfg['epochs']:
            logger.info(f"Epoch {epoch}/{cfg['epochs']} - 开始验证")
            val_psnr, val_ssim, val_loss = validate(model, val_loader, device, cfg, loss_fn)
            elapsed = time.time() - epoch_start
            logger.info(f"[Epoch {epoch}/{cfg['epochs']}] Train Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.4f} dB, Val SSIM: {val_ssim:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

            # 早停检查
            if cfg['early_stop']:
                should_stop = early_stopper(val_psnr)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    early_stopper.save_best_model(model, optimizer, epoch, os.path.join(exp_dir, 'best_model.pth'))
                    logger.info(f"保存最佳模型，Val PSNR: {val_psnr:.4f} dB")

                logger.info(f"早停状态 - 最佳PSNR: {early_stopper.best_score:.4f}, 停滞计数: {early_stopper.counter}/{cfg['early_stop_patience']}")

                if should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best val PSNR: {early_stopper.best_score:.4f} dB)")
                    break
            else:
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    early_stopper.save_best_model(model, optimizer, epoch, os.path.join(exp_dir, 'best_model.pth'))
                    logger.info(f"保存最佳模型，Val PSNR: {val_psnr:.4f} dB")
        else:
            elapsed = time.time() - epoch_start
            logger.info(f"[Epoch {epoch}/{cfg['epochs']}] Train Loss: {train_loss:.4f}, Time: {elapsed:.1f}s")

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"学习率更新: {current_lr:.6f}")

        # 写入 CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_l1:.6f}",
                f"{train_ssim:.4f}",
                f"{train_psnr:.4f}",
                f"{grad_norm:.4f}",
                f"{val_psnr:.4f}" if val_psnr is not None else '',
                f"{val_ssim:.4f}" if val_ssim is not None else '',
                f"{val_loss:.4f}" if val_loss is not None else '',
                f"{test_psnr:.4f}" if test_psnr is not None else '',
                f"{test_ssim:.4f}" if test_ssim is not None else '',
                f"{test_loss:.4f}" if test_loss is not None else '',
                f"{current_lr:.6f}"
            ])

        # 定期保存 checkpoint
        if epoch % cfg['save_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}.pth'))
            logger.info(f"保存 checkpoint: epoch_{epoch}.pth")

    logger.info(f"Training finished. Best val PSNR: {best_psnr:.4f} dB")


if __name__ == '__main__':
    main()
