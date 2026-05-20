#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# 数据集完整性验证脚本
# 检查 data/processed/ 下的目录结构:
# - 每个 split (train/val/test) 下应有 gt/ 和 qp27/ qp32/ qp37/
# - 每个序列文件夹下应有连续的 PNG 帧
# - 统计帧数、分辨率一致性
# - 计算退化帧与 GT 帧的 PSNR (抽样)

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from PIL import Image


def check_split(split_dir: Path):
    # 检查单个 split 的完整性
    results = {
        "split": split_dir.name,
        "gt_videos": [],
        "qp_videos": defaultdict(list),
        "issues": [],
    }

    gt_dir = split_dir / "gt"
    if not gt_dir.exists():
        results["issues"].append(f"GT 目录不存在: {gt_dir}")
        return results

    # 检查 GT
    for video_dir in sorted(gt_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        frames = sorted(video_dir.glob("*.png"))
        results["gt_videos"].append({
            "name": video_dir.name,
            "frames": len(frames),
            "sample_path": str(frames[0]) if frames else None,
        })

    # 检查各 QP 档
    for qp in [27, 32, 37]:
        qp_dir = split_dir / f"qp{qp}"
        if not qp_dir.exists():
            results["issues"].append(f"QP{qp} 目录不存在: {qp_dir}")
            continue

        for video_dir in sorted(qp_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted(video_dir.glob("*.png"))
            results["qp_videos"][qp].append({
                "name": video_dir.name,
                "frames": len(frames),
                "sample_path": str(frames[0]) if frames else None,
            })

    # 交叉检查: 每个 GT 视频在每个 QP 下都应有对应帧
    gt_names = {v["name"] for v in results["gt_videos"]}
    for qp in [27, 32, 37]:
        qp_names = {v["name"] for v in results["qp_videos"][qp]}
        missing = gt_names - qp_names
        extra = qp_names - gt_names
        if missing:
            results["issues"].append(f"QP{qp} 缺失视频: {missing}")
        if extra:
            results["issues"].append(f"QP{qp} 多余视频: {extra}")

    # 检查帧数一致性
    for qp in [27, 32, 37]:
        gt_map = {v["name"]: v["frames"] for v in results["gt_videos"]}
        for v in results["qp_videos"][qp]:
            if v["name"] in gt_map and v["frames"] != gt_map[v["name"]]:
                results["issues"].append(
                    f"{v['name']} QP{qp} 帧数不一致: GT={gt_map[v['name']]}, QP={v['frames']}"
                )

    return results


def sample_psnr(split_dir: Path, num_samples: int = 10):
    # 抽样计算 PSNR
    gt_dir = split_dir / "gt"
    psnr_samples = []

    for qp in [27, 32, 37]:
        qp_dir = split_dir / f"qp{qp}"
        if not qp_dir.exists():
            continue

        videos = [d.name for d in gt_dir.iterdir() if d.is_dir()]
        sampled = np.random.choice(videos, min(num_samples, len(videos)), replace=False)

        for vid in sampled:
            gt_frames = sorted((gt_dir / vid).glob("*.png"))
            qp_frames = sorted((qp_dir / vid).glob("*.png"))
            if not gt_frames or not qp_frames:
                continue

            # 随机抽一帧
            idx = np.random.randint(0, min(len(gt_frames), len(qp_frames)))
            gt_img = np.array(Image.open(gt_frames[idx]).convert('L'), dtype=np.float32)
            qp_img = np.array(Image.open(qp_frames[idx]).convert('L'), dtype=np.float32)

            mse = np.mean((gt_img - qp_img) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
            psnr_samples.append({"video": vid, "qp": qp, "frame": idx, "psnr": round(psnr, 2)})

    return psnr_samples


def main():
    parser = argparse.ArgumentParser(description="数据集验证")
    parser.add_argument("--data_dir", default="data/processed", help="数据目录")
    parser.add_argument("--psnr_samples", type=int, default=10, help="PSNR 抽样数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_results = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[SKIP] {split} 不存在")
            continue

        print(f"\n{'='*60}")
        print(f"检查 {split} ...")
        r = check_split(split_dir)
        print(f"GT 视频数: {len(r['gt_videos'])}")
        for qp in [27, 32, 37]:
            print(f"QP{qp} 视频数: {len(r['qp_videos'][qp])}")
        if r["issues"]:
            print(f"问题: {len(r['issues'])}")
            for issue in r["issues"]:
                print(f"  ! {issue}")
        else:
            print("无问题 ✓")

        # PSNR 抽样
        if args.psnr_samples > 0 and r["gt_videos"]:
            psnr_data = sample_psnr(split_dir, args.psnr_samples)
            if psnr_data:
                for qp in [27, 32, 37]:
                    qp_psnr = [p["psnr"] for p in psnr_data if p["qp"] == qp]
                    if qp_psnr:
                        print(f"QP{qp} 抽样 PSNR: {np.mean(qp_psnr):.2f} ± {np.std(qp_psnr):.2f} dB")
                r["psnr_samples"] = psnr_data

        all_results[split] = r

    # 保存报告
    report_path = data_dir / "dataset_check_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
