#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# 冒烟测试：验证 A/B/C 模型实例化、参数量、forward/backward、显存、损失、早停

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import CONFIG
from models import PureResUNet
from losses import L1SSIMLoss
from utils import EarlyStopping, calc_psnr


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_instantiation():
    print("=" * 60)
    print("[1] 模型实例化测试")
    print("=" * 60)

    cfg = CONFIG.copy()
    cfg['base_ch'] = 32

    model = PureResUNet(base_ch=cfg['base_ch'])
    n_params = count_params(model)
    print(f"  Model A: {n_params / 1e6:.2f}M params")

    # 1-batch forward
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 256, 256), "Model A output shape mismatch"
    print(f"  Model A: forward OK, output shape {out.shape}")

    print("[PASS] A 方案实例化与 forward 正常\n")


def test_param_budget():
    print("=" * 60)
    print("[2] 参数量预算测试 (base_ch=32 目标 5-15M)")
    print("=" * 60)

    model_a = PureResUNet(base_ch=32)
    n_params = count_params(model_a)
    print(f"  PureResUNet(base_ch=32): {n_params / 1e6:.2f}M params")

    assert 5e6 <= n_params <= 15e6, f"参数量 {n_params/1e6:.2f}M 不在 5-15M 范围内"
    print("[PASS] 参数量在预算范围内\n")


def test_forward_backward():
    print("=" * 60)
    print("[3] Forward + Backward + 显存测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PureResUNet(base_ch=32).to(device)
    loss_fn = L1SSIMLoss().to(device)

    # 1-batch
    x = torch.randn(1, 3, 256, 256, device=device)
    target = torch.randn(1, 3, 256, 256, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**2

    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  1-batch peak memory: {mem_peak:.2f} MB")
        assert mem_peak < 14 * 1024, f"显存峰值 {mem_peak:.2f}MB 超过 14GB"

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss 为 NaN 或 Inf"
    print(f"  Loss value: {loss.item():.4f}")
    print("[PASS] Forward + Backward 正常，显存 OK\n")

    # batch_size=8 test
    print("  Testing batch_size=8 ...")
    x8 = torch.randn(8, 3, 256, 256, device=device)
    target8 = torch.randn(8, 3, 256, 256, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    pred8 = model(x8)
    loss8 = loss_fn(pred8, target8)
    loss8.backward()

    if torch.cuda.is_available():
        mem_peak8 = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  8-batch peak memory: {mem_peak8:.2f} MB")
        assert mem_peak8 < 14 * 1024, f"batch=8 显存峰值 {mem_peak8:.2f}MB 超过 14GB"

    print("[PASS] batch_size=8 显存 OK\n")


def test_loss_fn():
    print("=" * 60)
    print("[4] 损失函数测试")
    print("=" * 60)

    loss_fn = L1SSIMLoss(l1_weight=1.0, ssim_weight=1.0)
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    loss = loss_fn(pred, target)
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss 异常"
    assert 0 <= loss.item() <= 10, f"Loss 数值不合理: {loss.item()}"
    print(f"  L1_SSIM Loss: {loss.item():.4f}")

    # 完美预测时 loss 应接近 0
    loss_zero = loss_fn(target, target)
    print(f"  Perfect prediction loss: {loss_zero.item():.6f}")
    assert loss_zero.item() < 0.01, "完美预测时 loss 应接近 0"

    print("[PASS] 损失函数数值合理\n")


def test_early_stopping():
    print("=" * 60)
    print("[5] 早停机制测试")
    print("=" * 60)

    es = EarlyStopping(patience=3, min_delta=0.1, mode='max')

    scores = [30.0, 30.05, 30.1, 30.15, 30.2, 30.25]
    stopped = False
    for i, score in enumerate(scores):
        should_stop = es(score)
        print(f"  Epoch {i}: val_psnr={score:.2f}, best={es.best_score:.2f}, counter={es.counter}")
        if should_stop:
            print(f"  Early stopping triggered at epoch {i}")
            stopped = True
            break

    assert not stopped, "不应在持续上升时触发早停"
    print("  [OK] 持续上升未触发早停")

    # 模拟不提升
    es2 = EarlyStopping(patience=3, min_delta=0.1, mode='max')
    flat_scores = [30.0, 30.0, 30.0, 30.0, 30.0]
    stopped2 = False
    for i, score in enumerate(flat_scores):
        should_stop = es2(score)
        print(f"  Epoch {i}: val_psnr={score:.2f}, best={es2.best_score:.2f}, counter={es2.counter}, stop={should_stop}")
        if should_stop:
            stopped2 = True
            break

    assert stopped2, "应在连续不提升时触发早停"
    assert es2.counter >= 3, "counter 应达到 patience"
    print("  [OK] 连续不提升触发早停")

    print("[PASS] 早停机制正常\n")


def test_psnr_metric():
    print("=" * 60)
    print("[6] PSNR 指标测试")
    print("=" * 60)

    pred = torch.ones(1, 3, 64, 64)
    target = torch.ones(1, 3, 64, 64)
    psnr = calc_psnr(pred, target)
    assert psnr > 100, f"完全相同图像 PSNR 应极大，实际 {psnr}"
    print(f"  Identical PSNR: {psnr:.2f} dB")

    pred2 = torch.ones(1, 3, 64, 64) * 0.5
    target2 = torch.ones(1, 3, 64, 64)
    psnr2 = calc_psnr(pred2, target2)
    print(f"  MSE=0.25 PSNR: {psnr2:.2f} dB")
    # 数据范围 [0,1]，MSE=0.25 时 PSNR = 10*log10(1/0.25) = 6.02 dB
    assert 4 < psnr2 < 10, f"MSE=0.25 时 PSNR 应在 4-10 之间，实际 {psnr2}"

    print("[PASS] PSNR 指标正常\n")


if __name__ == '__main__':
    test_model_instantiation()
    test_param_budget()
    test_forward_backward()
    test_loss_fn()
    test_early_stopping()
    test_psnr_metric()
    print("=" * 60)
    print("所有冒烟测试通过！")
    print("=" * 60)
