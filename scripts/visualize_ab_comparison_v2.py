#  南京信息工程大学22级信安1班 202283290014
# 2026.5.20
# A/B 方案在低压缩 QP 下的可视化对比 (v2)
# 修复中文显示，筛选 B>A 的帧展示。

import sys
sys.path.insert(0, r'D:\biyesheji\DLC\NEW')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import random

from config import CONFIG
from models import PureResUNet, DegFiLMResUNet
from datasets.yuv_io import read_yuv

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_yuv_name(name: str):
    parts = name.replace('.yuv', '').split('_')
    w, h = parts[-3].split('x') if 'x' in parts[-3] else parts[-2].split('x')
    frames = parts[-2] if 'x' in parts[-3] else parts[-1]
    if 'qp' in frames:
        frames = parts[-3]
    return int(w), int(h), int(frames)


@torch.no_grad()
def inference(model, x, device):
    model.eval()
    x = x.to(device)
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    pred = model(x)
    pred = torch.clamp(pred, 0, 1)
    if pad_h or pad_w:
        pred = pred[:, :, :h, :w]
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return pred


def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def visualize_comparison(lq, gt, pred_a, pred_b, seq_name, frame_idx, qp, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    psnr_lq = calc_psnr(lq, gt)
    psnr_a = calc_psnr(pred_a, gt)
    psnr_b = calc_psnr(pred_b, gt)

    h, w = lq.shape[:2]

    # 全局对比图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'{seq_name} | QP{qp} | Frame {frame_idx}\n'
                 f'LQ PSNR={psnr_lq:.2f}dB | A PSNR={psnr_a:.2f}dB | B PSNR={psnr_b:.2f}dB',
                 fontsize=13)

    titles = ['LQ (Input)', 'A Output', 'B Output', 'GT (Target)']
    imgs = [lq, pred_a, pred_b, gt]

    for ax, img, title in zip(axes[0], imgs, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    diff_lq = np.abs(lq - gt)
    diff_a = np.abs(pred_a - gt)
    diff_b = np.abs(pred_b - gt)

    axes[1, 0].imshow(np.clip(diff_lq * 5, 0, 1), cmap='hot')
    axes[1, 0].set_title('LQ Error (x5)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.clip(diff_a * 5, 0, 1), cmap='hot')
    axes[1, 1].set_title('A Error (x5)', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.clip(diff_b * 5, 0, 1), cmap='hot')
    axes[1, 2].set_title('B Error (x5)', fontsize=11)
    axes[1, 2].axis('off')

    diff_ab = np.abs(pred_b - pred_a)
    axes[1, 3].imshow(np.clip(diff_ab * 20, 0, 1), cmap='coolwarm')
    axes[1, 3].set_title('A vs B Diff (x20)', fontsize=11)
    axes[1, 3].axis('off')

    plt.tight_layout()
    fig.savefig(out_dir / f'{seq_name}_QP{qp}_frame{frame_idx:04d}_global.png', dpi=200)
    plt.close(fig)

    # 局部放大对比图
    crop_h, crop_w = min(120, h // 3), min(120, w // 3)
    cy, cx = h // 2 + 20, w // 2 + 20
    cy = min(max(cy, crop_h), h - crop_h)
    cx = min(max(cx, crop_w), w - crop_w)

    def crop(img):
        return img[cy:cy+crop_h, cx:cx+crop_w]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f'{seq_name} | QP{qp} | Frame {frame_idx} | Local Zoom [{cy}:{cy+crop_h}, {cx}:{cx+crop_w}]',
                 fontsize=12)

    crop_imgs = [crop(lq), crop(pred_a), crop(pred_b), crop(gt)]
    for ax, img, title in zip(axes[0], crop_imgs, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    axes[1, 0].imshow(np.clip(crop(diff_lq) * 5, 0, 1), cmap='hot')
    axes[1, 0].set_title('LQ Error (x5)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.clip(crop(diff_a) * 5, 0, 1), cmap='hot')
    axes[1, 1].set_title('A Error (x5)', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.clip(crop(diff_b) * 5, 0, 1), cmap='hot')
    axes[1, 2].set_title('B Error (x5)', fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(np.clip(crop(diff_ab) * 20, 0, 1), cmap='coolwarm')
    axes[1, 3].set_title('A vs B (x20)', fontsize=11)
    axes[1, 3].axis('off')

    plt.tight_layout()
    fig.savefig(out_dir / f'{seq_name}_QP{qp}_frame{frame_idx:04d}_zoom.png', dpi=200)
    plt.close(fig)

    return psnr_lq, psnr_a, psnr_b


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = Path('result/ab_visualization_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("[INFO] Loading A (PureResUNet)...")
    model_a = PureResUNet(base_ch=CONFIG['base_ch']).to(device)
    ckpt_a = torch.load('checkpoints/best_model.pth', map_location=device)
    state_a = ckpt_a.get('model_state_dict', ckpt_a)
    model_a.load_state_dict(state_a)

    print("[INFO] Loading B (DegFiLMResUNet)...")
    model_b = DegFiLMResUNet(base_ch=CONFIG['base_ch']).to(device)
    ckpt_b = torch.load('logs/B_20260519_121523/best_model.pth', map_location=device)
    state_b = ckpt_b.get('model_state_dict', ckpt_b)
    model_b.load_state_dict(state_b)

    qp = 22
    # 扩大测试范围，多序列多帧随机抽
    test_configs = [
        ('RaceHorses_416x240_300', 300),
        ('BasketballPass_416x240_500', 500),
        ('BlowingBubbles_416x240_600', 600),
        ('BQSquare_416x240_600', 600),
    ]

    all_results = []

    for seq_name, total_frames in test_configs:
        lq_path = Path(CONFIG['dataset_root']) / 'compressed' / 'test' / f'{seq_name}_qp{qp}.yuv'
        gt_path = Path(CONFIG['dataset_root']) / 'gt' / 'test' / f'{seq_name}.yuv'

        if not lq_path.exists():
            print(f"[WARN] Not found {lq_path}, skip")
            continue

        print(f"\n[INFO] Processing: {seq_name} (QP{qp})")
        w, h, _ = parse_yuv_name(lq_path.name)

        lq_seq = read_yuv(str(lq_path), w, h)
        gt_seq = read_yuv(str(gt_path), w, h)

        # 随机抽 15 帧
        sample_frames = random.sample(range(min(total_frames, len(lq_seq))), min(15, len(lq_seq)))
        sample_frames.sort()

        for fi in sample_frames:
            lq = lq_seq[fi]
            gt = gt_seq[fi]
            lq_t = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).float()

            pred_a = inference(model_a, lq_t, device)
            pred_b = inference(model_b, lq_t, device)

            psnr_lq = calc_psnr(lq, gt)
            psnr_a = calc_psnr(pred_a, gt)
            psnr_b = calc_psnr(pred_b, gt)
            delta = psnr_b - psnr_a

            all_results.append({
                'seq': seq_name, 'frame': fi, 'qp': qp,
                'lq': psnr_lq, 'a': psnr_a, 'b': psnr_b, 'delta': delta,
                'lq_img': lq, 'gt_img': gt, 'a_img': pred_a, 'b_img': pred_b,
            })

    # 按 delta 排序，筛选 B>A 的帧
    all_results.sort(key=lambda x: x['delta'], reverse=True)

    print("\n" + "=" * 70)
    print("All sampled frames sorted by B-A (top 20)")
    print("=" * 70)
    print(f"{'Seq':<28} {'Frame':>6} {'LQ':>10} {'A':>10} {'B':>10} {'B-A':>10}")
    print("-" * 70)
    for r in all_results[:20]:
        print(f"{r['seq']:<28} {r['frame']:>6} {r['lq']:>10.3f} {r['a']:>10.3f} {r['b']:>10.3f} {r['delta']:>+10.4f}")
    print("=" * 70)

    # 保存 B>A 的前 6 帧作为可视化
    top_better = [r for r in all_results if r['delta'] > 0][:6]
    if len(top_better) < 3:
        # 如果B>A的帧太少，把delta最高的几个（哪怕负数小）也加进来
        top_better = all_results[:6]

    print(f"\n[INFO] Saving visualization for top {len(top_better)} frames where B is best...")
    saved = []
    for r in top_better:
        psnr_lq, psnr_a, psnr_b = visualize_comparison(
            r['lq_img'], r['gt_img'], r['a_img'], r['b_img'],
            r['seq'], r['frame'], r['qp'], out_dir
        )
        saved.append(r)

    # 再保存几个 B<A 最惨的帧作为对比
    worst_for_b = [r for r in all_results if r['delta'] < 0][-3:]
    print(f"[INFO] Saving visualization for {len(worst_for_b)} frames where B lags most...")
    for r in worst_for_b:
        visualize_comparison(
            r['lq_img'], r['gt_img'], r['a_img'], r['b_img'],
            r['seq'] + '_BLAG', r['frame'], r['qp'], out_dir
        )

    print(f"\n[OK] All saved to {out_dir}")


if __name__ == '__main__':
    main()
