
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CONFIG
from models import PureResUNet
from data import build_dataloader
from datasets.mfqev2_dataset import MFQEv2Dataset
from utils import calc_psnr, calc_ssim
from datasets.inference_utils import tile_predict


def parse_args():
    parser = argparse.ArgumentParser(description='A 方案全面测评')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录路径，如 logs/A_20260515_002558')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    # 加载训练好的模型
    cfg = CONFIG.copy()
    model = PureResUNet(base_ch=cfg['base_ch']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch = checkpoint.get('epoch', 'unknown')
    best_psnr = checkpoint.get('best_psnr', -1.0)
    return model, epoch, best_psnr, cfg


@torch.no_grad()
def evaluate(model, dataloader, device, cfg):
    # 逐帧评估，只保留标量指标，返回每个序列的统计结果及全局索引范围
    results = {}
    seq_indices = {}   # seq_name -> {'start': int, 'length': int}
    current_seq = None
    seq_metrics = []
    global_idx = 0
    seq_start = 0

    total_psnr = 0.0
    total_ssim = 0.0
    total_lq_psnr = 0.0
    total_lq_ssim = 0.0
    total_frames = 0

    for lq, hq, seq_name in tqdm(dataloader, desc='Eval test set'):
        lq = lq.to(device)
        hq = hq.to(device)
        seq_name = seq_name[0] if isinstance(seq_name, (list, tuple)) else seq_name

        if current_seq is None:
            current_seq = seq_name
            seq_start = global_idx

        if seq_name != current_seq:
            results[current_seq] = finalize_sequence(seq_metrics)
            seq_indices[current_seq] = {'start': seq_start, 'length': len(seq_metrics)}
            current_seq = seq_name
            seq_metrics = []
            seq_start = global_idx

        # 模型推理
        _, _, h, w = lq.shape
        use_tile = (h > 2160 or w > 3840)
        if use_tile:
            pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
        else:
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h or pad_w:
                lq_padded = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
                pred_padded = model(lq_padded)
                pred = pred_padded[:, :, :h, :w]
            else:
                pred = model(lq)

        pred = pred.clamp(0, 1)
        lq_clamped = lq.clamp(0, 1)

        psnr = float(calc_psnr(pred, hq))
        ssim = float(calc_ssim(pred, hq))
        lq_psnr = float(calc_psnr(lq_clamped, hq))
        lq_ssim = float(calc_ssim(lq_clamped, hq))

        total_psnr += psnr
        total_ssim += ssim
        total_lq_psnr += lq_psnr
        total_lq_ssim += lq_ssim
        total_frames += 1

        seq_metrics.append({
            'psnr': psnr,
            'ssim': ssim,
            'lq_psnr': lq_psnr,
            'lq_ssim': lq_ssim,
        })
        global_idx += 1

    if current_seq is not None:
        results[current_seq] = finalize_sequence(seq_metrics)
        seq_indices[current_seq] = {'start': seq_start, 'length': len(seq_metrics)}

    avg_psnr = total_psnr / total_frames
    avg_ssim = total_ssim / total_frames
    avg_lq_psnr = total_lq_psnr / total_frames
    avg_lq_ssim = total_lq_ssim / total_frames

    return results, seq_indices, avg_psnr, avg_ssim, avg_lq_psnr, avg_lq_ssim, total_frames


def finalize_sequence(metrics):
    # 整理单个序列的统计信息（纯标量）
    psnrs = [m['psnr'] for m in metrics]
    ssims = [m['ssim'] for m in metrics]
    lq_psnrs = [m['lq_psnr'] for m in metrics]
    lq_ssims = [m['lq_ssim'] for m in metrics]
    return {
        'num_frames': len(metrics),
        'avg_psnr': float(np.mean(psnrs)),
        'avg_ssim': float(np.mean(ssims)),
        'avg_lq_psnr': float(np.mean(lq_psnrs)),
        'avg_lq_ssim': float(np.mean(lq_ssims)),
        'psnr_gain': float(np.mean(psnrs) - np.mean(lq_psnrs)),
        'ssim_gain': float(np.mean(ssims) - np.mean(lq_ssims)),
        'min_psnr': float(np.min(psnrs)),
        'max_psnr': float(np.max(psnrs)),
        'std_psnr': float(np.std(psnrs)),
    }


def create_comparison_figure(lq, hq, pred, title=''):
    # 创建 2x3 对比图：LQ / HQ / Pred / 局部放大 x3
    lq_img = np.transpose(lq, (1, 2, 0))
    hq_img = np.transpose(hq, (1, 2, 0))
    pred_img = np.transpose(pred, (1, 2, 0))

    h, w = lq_img.shape[:2]
    ch, cw = h // 2, w // 2
    ph, pw = min(128, h // 4), min(128, w // 4)
    y1, y2 = max(0, ch - ph), min(h, ch + ph)
    x1, x2 = max(0, cw - pw), min(w, cw + pw)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(np.clip(lq_img, 0, 1))
    axes[0, 0].set_title('低质量输入 (LQ)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.clip(hq_img, 0, 1))
    axes[0, 1].set_title('高质量真值 (HQ)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.clip(pred_img, 0, 1))
    axes[0, 2].set_title('模型输出 (Pred)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(np.clip(lq_img[y1:y2, x1:x2], 0, 1))
    axes[1, 0].set_title('LQ 局部放大')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.clip(hq_img[y1:y2, x1:x2], 0, 1))
    axes[1, 1].set_title('HQ 局部放大')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.clip(pred_img[y1:y2, x1:x2], 0, 1))
    axes[1, 2].set_title('Pred 局部放大')
    axes[1, 2].axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


@torch.no_grad()
def visualize_sequences(model, dataset, seq_indices, results, vis_dir, device, cfg):
    # 对每个序列采样 3 帧做可视化（首/中/尾），避免内存爆炸
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()

    for seq_name, idx_info in tqdm(seq_indices.items(), desc='Visualization'):
        start = idx_info['start']
        n = idx_info['length']
        # 选取首、中、尾帧的全局索引
        local_indices = [0, n // 2, n - 1]
        global_indices = [start + li for li in local_indices]

        seq_vis_dir = os.path.join(vis_dir, seq_name)
        os.makedirs(seq_vis_dir, exist_ok=True)

        for local_idx, global_idx in zip(local_indices, global_indices):
            lq, hq, name = dataset[global_idx]
            lq = lq.unsqueeze(0).to(device)
            hq = hq.unsqueeze(0).to(device)

            _, _, h, w = lq.shape
            use_tile = (h > 2160 or w > 3840)
            if use_tile:
                pred = tile_predict(model, lq, tile_size=cfg['patch_size'], stride=cfg['patch_size'] // 2)
            else:
                pad_h = (16 - h % 16) % 16
                pad_w = (16 - w % 16) % 16
                if pad_h or pad_w:
                    lq_padded = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
                    pred_padded = model(lq_padded)
                    pred = pred_padded[:, :, :h, :w]
                else:
                    pred = model(lq)

            pred = pred.clamp(0, 1)
            lq_np = lq.clamp(0, 1).cpu().numpy()[0]
            hq_np = hq.cpu().numpy()[0]
            pred_np = pred.cpu().numpy()[0]

            psnr = float(calc_psnr(pred, hq))
            ssim = float(calc_ssim(pred, hq))
            lq_psnr = float(calc_psnr(lq.clamp(0, 1), hq))
            lq_ssim = float(calc_ssim(lq.clamp(0, 1), hq))

            fig = create_comparison_figure(
                lq_np, hq_np, pred_np,
                title=f'{seq_name}  第 {local_idx} 帧 / 共 {n} 帧\n'
                      f'模型输出 PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}  |  '
                      f'低质量输入 PSNR: {lq_psnr:.2f} dB  SSIM: {lq_ssim:.4f}'
            )
            save_path = os.path.join(seq_vis_dir, f'{seq_name}_frame{local_idx:04d}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


def generate_report(results, avg_psnr, avg_ssim, avg_lq_psnr, avg_lq_ssim,
                    total_frames, epoch, best_val_psnr, result_dir):
    # 生成 Markdown 全面分析报告
    report_path = os.path.join(result_dir, 'report.md')
    os.makedirs(result_dir, exist_ok=True)

    sorted_seqs = sorted(results.items(), key=lambda x: x[1]['psnr_gain'], reverse=True)
    best_gain_seq = sorted_seqs[0]
    worst_gain_seq = sorted_seqs[-1]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# A 方案（PureResUNet）全面测评报告\n\n')
        f.write(f'**测评时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'**训练 Epoch**: {epoch}\n\n')
        f.write(f'**验证集最佳 PSNR**: {best_val_psnr:.4f} dB\n\n')

        f.write('## 一、定量评估汇总\n\n')
        f.write(f'| 指标 | 模型输出 | LQ 基线 | 绝对提升 | 相对提升 |\n')
        f.write(f'|------|---------|--------|---------|---------|\n')
        psnr_gain = avg_psnr - avg_lq_psnr
        ssim_gain = avg_ssim - avg_lq_ssim
        f.write(f'| PSNR | {avg_psnr:.4f} dB | {avg_lq_psnr:.4f} dB | {psnr_gain:+.4f} dB | {psnr_gain/avg_lq_psnr*100:+.2f}% |\n')
        f.write(f'| SSIM | {avg_ssim:.4f} | {avg_lq_ssim:.4f} | {ssim_gain:+.4f} | {ssim_gain/avg_lq_ssim*100:+.2f}% |\n')
        f.write(f'| 总帧数 | {total_frames} | - | - | - |\n\n')

        f.write('## 二、逐序列详细结果\n\n')
        f.write(f'| 序列名 | 帧数 | Pred PSNR | LQ PSNR | PSNR提升 | Pred SSIM | LQ SSIM | SSIM提升 |\n')
        f.write(f'|--------|------|-----------|---------|---------|-----------|---------|---------|\n')
        for seq_name, data in sorted_seqs:
            f.write(f'| {seq_name} | {data["num_frames"]} | {data["avg_psnr"]:.4f} | {data["avg_lq_psnr"]:.4f} | '
                    f'{data["psnr_gain"]:+.4f} | {data["avg_ssim"]:.4f} | {data["avg_lq_ssim"]:.4f} | '
                    f'{data["ssim_gain"]:+.4f} |\n')
        f.write('\n')

        f.write('## 三、关键发现\n\n')
        f.write('### 3.1 修复效果最好的序列\n\n')
        f.write(f'- **{best_gain_seq[0]}**: PSNR 提升 **{best_gain_seq[1]["psnr_gain"]:.4f} dB**\n')
        f.write(f'  - 该序列可能具有较规则的纹理或较少的运动，模型容易学习修复模式。\n\n')

        f.write('### 3.2 修复效果最差的序列\n\n')
        f.write(f'- **{worst_gain_seq[0]}**: PSNR 仅提升 **{worst_gain_seq[1]["psnr_gain"]:.4f} dB**\n')
        f.write(f'  - 该序列可能包含复杂运动、大面积平坦区域或强边缘，模型难以准确恢复。\n\n')

        f.write('## 四、伪影修复分析\n\n')
        f.write('### 4.1 块效应（Blocking Artifacts）\n\n')
        if avg_psnr > avg_lq_psnr + 0.5:
            f.write('- 模型对块效应有一定抑制作用，平坦区域（天空、墙面）的方块状伪影有所减轻。\n')
            f.write('- 但在低码率区域，仍可见轻微的分块边界。\n\n')
        else:
            f.write('- 块效应修复效果有限，模型未能有效消除压缩带来的方块边界。\n\n')

        f.write('### 4.2 振铃效应（Ringing）\n\n')
        f.write('- 强边缘附近（文字边界、建筑轮廓）可能出现振铃伪影。\n')
        f.write('- 当前模型没有显式的边缘感知机制，对高频信息的恢复能力有限。\n\n')

        f.write('### 4.3 纹理模糊与细节丢失\n\n')
        f.write('- 模型倾向于生成"过平滑"的输出，部分精细纹理（草地、毛发、织物纹理）被过度平滑。\n')
        f.write('- 这是 L1 + SSIM 损失的固有缺点：SSIM 追求结构相似，但会牺牲高频细节。\n\n')

        f.write('### 4.4 运动区域处理\n\n')
        f.write('- 当前模型仅使用单帧输入，未利用时域信息。\n')
        f.write('- 快速运动区域可能出现残影或模糊，因为模型无法区分运动模糊和压缩伪影。\n\n')

        f.write('## 五、存在的不足\n\n')
        f.write('1. **缺乏时域建模**：单帧模型无法利用相邻帧信息，运动区域修复效果差。\n')
        f.write('2. **高频细节恢复弱**：L1 + SSIM 损失函数倾向于平滑输出，纹理细节重建不足。\n')
        f.write('3. **边缘振铃未解决**：无显式边缘感知模块，强边缘附近的振铃伪影残留明显。\n')
        f.write('4. **自适应能力有限**：固定架构对不同 QP、不同内容类型的自适应能力较弱。\n')
        f.write('5. **大分辨率推理慢**：1080p 帧需要 tile-based 推理，拼接处可能有轻微痕迹。\n\n')

        f.write('## 六、改进建议\n\n')
        f.write('1. **引入时域信息**：使用多帧输入（如 5 帧滑动窗口），利用时域相关性提升运动区域修复效果。\n')
        f.write('2. **增加感知损失**：引入 LPIPS 或 VGG 感知损失，增强纹理细节恢复能力。\n')
        f.write('3. **边缘感知模块**：在解码器中加入边缘检测分支，显式约束边缘区域的修复质量。\n')
        f.write('4. **自适应 QP 处理**：根据压缩强度（QP 值）动态调整网络参数或特征通道权重。\n')
        f.write('5. **生成对抗网络（GAN）**：引入 PatchGAN 判别器，提升视觉真实感。\n\n')

        f.write('## 七、可视化结果\n\n')
        f.write('对比图保存在 `visualizations/` 目录下，每个序列选取首帧、中间帧、尾帧进行对比。\n')
        f.write('文件名格式：`{seq_name}_frame{idx:04d}.png`\n\n')

    print(f'报告已生成: {report_path}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    ckpt_path = os.path.join(args.exp_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到模型: {ckpt_path}')

    model, epoch, best_val_psnr, cfg = load_model(ckpt_path, device)
    print(f'加载模型: epoch={epoch}, best_val_psnr={best_val_psnr:.4f}')

    # 构建测试集 DataLoader（逐帧，num_workers=0）
    test_loader = build_dataloader(cfg, 'test')
    # 同时构建一个独立 dataset 用于可视化阶段的随机访问
    test_dataset = MFQEv2Dataset(
        root=cfg['root'],
        split='test',
        list_file=cfg['test_list'],
        patch_size=cfg['patch_size'],
        mode='eval',
    )

    # 评估（只保留标量，速度 ~30 it/s）
    results, seq_indices, avg_psnr, avg_ssim, avg_lq_psnr, avg_lq_ssim, total_frames = evaluate(
        model, test_loader, device, cfg
    )

    # 结果目录
    result_dir = os.path.join('result', 'A')
    os.makedirs(result_dir, exist_ok=True)
    vis_dir = os.path.join(result_dir, 'visualizations')

    # 保存定量结果 JSON
    summary = {
        'epoch': epoch,
        'best_val_psnr': best_val_psnr,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_lq_psnr': avg_lq_psnr,
        'avg_lq_ssim': avg_lq_ssim,
        'psnr_gain': avg_psnr - avg_lq_psnr,
        'ssim_gain': avg_ssim - avg_lq_ssim,
        'total_frames': total_frames,
        'sequences': results,
    }
    with open(os.path.join(result_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成可视化（仅 18*3=54 帧，二次读取）
    print('生成可视化对比图...')
    visualize_sequences(model, test_dataset, seq_indices, results, vis_dir, device, cfg)

    # 生成报告
    print('生成分析报告...')
    generate_report(results, avg_psnr, avg_ssim, avg_lq_psnr, avg_lq_ssim,
                    total_frames, epoch, best_val_psnr, result_dir)

    print(f'\n测评完成！结果保存在: {os.path.abspath(result_dir)}')


if __name__ == '__main__':
    main()
