#  南京信息工程大学22级信安1班 202283290014
# 2026.5.20
# A/B 方案跨QP评估结果对比汇总
# 读取 A 方案和 B 方案的 cross_qp_results.json，生成对比图表和汇总表格。

import sys
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_psnr_comparison(a_res, b_res, out_dir):
    out_dir = Path(out_dir)
    qp_list = [r['qp'] for r in a_res]
    a_psnr = [r['psnr'] for r in a_res]
    b_psnr = [r['psnr'] for r in b_res]
    base_psnr = [r['baseline_psnr'] for r in a_res]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qp_list, base_psnr, marker='x', linewidth=2, markersize=8, color='#888888', linestyle='--', label='Baseline (输入LQ)')
    ax.plot(qp_list, a_psnr, marker='o', linewidth=2.5, markersize=9, color='#2E86AB', label='A方案 (PureResUNet)')
    ax.plot(qp_list, b_psnr, marker='s', linewidth=2.5, markersize=9, color='#E94F37', label='B方案 (DegFiLMResUNet)')

    ax.set_xlabel('QP（压缩强度）', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('A/B 方案跨QP PSNR 对比', fontsize=14)
    ax.set_xticks(qp_list)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    for x, y in zip(qp_list, a_psnr):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='#2E86AB')
    for x, y in zip(qp_list, b_psnr):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color='#E94F37')

    fig.tight_layout()
    fig.savefig(out_dir / 'ab_psnr_comparison.png', dpi=300)
    plt.close(fig)
    print(f"[OK] ab_psnr_comparison.png")


def plot_ssim_comparison(a_res, b_res, out_dir):
    out_dir = Path(out_dir)
    qp_list = [r['qp'] for r in a_res]
    a_ssim = [r['ssim'] for r in a_res]
    b_ssim = [r['ssim'] for r in b_res]
    base_ssim = [r['baseline_ssim'] for r in a_res]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qp_list, base_ssim, marker='x', linewidth=2, markersize=8, color='#888888', linestyle='--', label='Baseline (输入LQ)')
    ax.plot(qp_list, a_ssim, marker='o', linewidth=2.5, markersize=9, color='#2E86AB', label='A方案 (PureResUNet)')
    ax.plot(qp_list, b_ssim, marker='s', linewidth=2.5, markersize=9, color='#E94F37', label='B方案 (DegFiLMResUNet)')

    ax.set_xlabel('QP（压缩强度）', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('A/B 方案跨QP SSIM 对比', fontsize=14)
    ax.set_xticks(qp_list)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    for x, y in zip(qp_list, a_ssim):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='#2E86AB')
    for x, y in zip(qp_list, b_ssim):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color='#E94F37')

    fig.tight_layout()
    fig.savefig(out_dir / 'ab_ssim_comparison.png', dpi=300)
    plt.close(fig)
    print(f"[OK] ab_ssim_comparison.png")


def plot_gain_comparison(a_res, b_res, out_dir):
    out_dir = Path(out_dir)
    qp_list = [r['qp'] for r in a_res]
    a_delta_psnr = [r['delta_psnr'] for r in a_res]
    b_delta_psnr = [r['delta_psnr'] for r in b_res]
    a_delta_ssim = [r['delta_ssim'] for r in a_res]
    b_delta_ssim = [r['delta_ssim'] for r in b_res]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = '#2E86AB'
    ax1.set_xlabel('QP（压缩强度）', fontsize=12)
    ax1.set_ylabel('PSNR 增益 (dB)', color=color1, fontsize=12)
    ax1.plot(qp_list, a_delta_psnr, marker='o', linewidth=2.5, markersize=9, color=color1, label='A方案 PSNR增益')
    ax1.plot(qp_list, b_delta_psnr, marker='s', linewidth=2.5, markersize=9, color='#E94F37', label='B方案 PSNR增益')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(qp_list)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('SSIM 增益', color=color2, fontsize=12)
    ax2.plot(qp_list, a_delta_ssim, marker='o', linewidth=2, markersize=9, color=color2, linestyle='--', label='A方案 SSIM增益')
    ax2.plot(qp_list, b_delta_ssim, marker='s', linewidth=2, markersize=9, color='#F39C12', linestyle='--', label='B方案 SSIM增益')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('A/B 方案跨QP增益对比', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'ab_gain_comparison.png', dpi=300)
    plt.close(fig)
    print(f"[OK] ab_gain_comparison.png")


def save_summary_csv(a_res, b_res, out_dir):
    out_dir = Path(out_dir)
    rows = []
    for ra, rb in zip(sorted(a_res, key=lambda x: x['qp']), sorted(b_res, key=lambda x: x['qp'])):
        qp = ra['qp']
        rows.append({
            'qp': qp,
            'A_psnr': ra['psnr'],
            'B_psnr': rb['psnr'],
            'diff_psnr (B-A)': rb['psnr'] - ra['psnr'],
            'A_ssim': ra['ssim'],
            'B_ssim': rb['ssim'],
            'diff_ssim (B-A)': rb['ssim'] - ra['ssim'],
            'A_delta_psnr': ra['delta_psnr'],
            'B_delta_psnr': rb['delta_psnr'],
            'diff_gain_psnr (B-A)': rb['delta_psnr'] - ra['delta_psnr'],
            'A_delta_ssim': ra['delta_ssim'],
            'B_delta_ssim': rb['delta_ssim'],
            'diff_gain_ssim (B-A)': rb['delta_ssim'] - ra['delta_ssim'],
        })

    path = out_dir / 'ab_comparison_summary.csv'
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {path}")

    # 同时打印到控制台
    print("\n" + "=" * 100)
    print("A/B 方案跨QP对比汇总")
    print("=" * 100)
    print(f"{'QP':>4} {'A_PSNR':>10} {'B_PSNR':>10} {'ΔPSNR':>10} {'A_SSIM':>10} {'B_SSIM':>10} {'ΔSSIM':>10} {'A_GainP':>10} {'B_GainP':>10} {'ΔGainP':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['qp']:4d} {r['A_psnr']:10.4f} {r['B_psnr']:10.4f} {r['diff_psnr (B-A)']:+10.4f} "
              f"{r['A_ssim']:10.4f} {r['B_ssim']:10.4f} {r['diff_ssim (B-A)']:+10.4f} "
              f"{r['A_delta_psnr']:10.4f} {r['B_delta_psnr']:10.4f} {r['diff_gain_psnr (B-A)']:+10.4f}")
    print("=" * 100)


def main():
    a_path = Path('result/cross_qp/cross_qp_results.json')
    b_path = Path('result/cross_qp/B_20260519_121523/cross_qp_results.json')
    out_dir = Path('result/cross_qp/B_20260519_121523')

    if not a_path.exists():
        print(f"[ERROR] 未找到A方案结果: {a_path}")
        sys.exit(1)
    if not b_path.exists():
        print(f"[ERROR] 未找到B方案结果: {b_path}")
        sys.exit(1)

    ensure_dir(out_dir)
    a_res = load_results(a_path)
    b_res = load_results(b_path)

    plot_psnr_comparison(a_res, b_res, out_dir)
    plot_ssim_comparison(a_res, b_res, out_dir)
    plot_gain_comparison(a_res, b_res, out_dir)
    save_summary_csv(a_res, b_res, out_dir)

    print(f"\n[Done] 所有对比图表已保存至 {out_dir}")


if __name__ == '__main__':
    main()
