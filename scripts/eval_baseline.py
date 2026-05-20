import json
import torch
from tqdm import tqdm
import sys
import os

# 把项目根目录加入路径，确保能 import config/data/utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from data import build_dataloader
from utils import calc_psnr, calc_ssim


@torch.no_grad()
def eval_baseline_split(split='val'):
    # 对指定 split 做基线测评
    cfg = CONFIG.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = build_dataloader(cfg, split)

    total_psnr = 0.0
    total_ssim = 0.0
    total_frames = 0

    seq_results = {}
    current_seq = None
    seq_psnr = 0.0
    seq_ssim = 0.0
    seq_frames = 0

    for lq, hq, seq_name in tqdm(loader, desc=f'Baseline {split}', leave=False):
        lq = lq.to(device)   # (1, C, H, W)
        hq = hq.to(device)
        seq_name = seq_name[0] if isinstance(seq_name, (list, tuple)) else seq_name

        if current_seq is None:
            current_seq = seq_name

        if seq_name != current_seq:
            seq_results[current_seq] = {
                'psnr': round(seq_psnr / seq_frames, 4),
                'ssim': round(seq_ssim / seq_frames, 4),
                'frames': seq_frames
            }
            current_seq = seq_name
            seq_psnr = 0.0
            seq_ssim = 0.0
            seq_frames = 0

        psnr = calc_psnr(lq.clamp(0, 1), hq)
        ssim = calc_ssim(lq.clamp(0, 1), hq)

        seq_psnr += psnr
        seq_ssim += ssim
        seq_frames += 1
        total_psnr += psnr
        total_ssim += ssim
        total_frames += 1

    # 最后一个序列
    if current_seq is not None:
        seq_results[current_seq] = {
            'psnr': round(seq_psnr / seq_frames, 4),
            'ssim': round(seq_ssim / seq_frames, 4),
            'frames': seq_frames
        }

    avg_psnr = total_psnr / total_frames if total_frames > 0 else 0.0
    avg_ssim = total_ssim / total_frames if total_frames > 0 else 0.0

    return {
        'split': split,
        'total_frames': total_frames,
        'avg_psnr': round(avg_psnr, 4),
        'avg_ssim': round(avg_ssim, 4),
        'sequences': seq_results
    }


def main():
    results = {}
    for split in ['val', 'test']:
        result = eval_baseline_split(split)
        results[split] = result

        out_file = f'result/baseline/baseline_{split}.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[{split.upper()}] Baseline 完成")
        print(f"  总帧数: {result['total_frames']}")
        print(f"  平均 PSNR: {result['avg_psnr']:.4f} dB")
        print(f"  平均 SSIM: {result['avg_ssim']:.4f}")
        print(f"  结果已保存: {out_file}")
        print()

    # 同时保存合并摘要
    with open('result/baseline/baseline_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("合并摘要已保存: result/baseline/baseline_summary.json")


if __name__ == '__main__':
    main()
