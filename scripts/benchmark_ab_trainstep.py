#  南京信息工程大学22级信安1班 202283290014
# 2026.5.18 (updated: WaveletResUNet -> DegFiLMResUNet)
# A/B 方案训练步骤性能对比（含 backward）
# 检测 B 方案是否存在 backward 瓶颈、CPU 占用异常

import sys
sys.path.insert(0, r'D:\biyesheji\DLC\NEW')

import torch
import time
import psutil
import os
from models.pure_resunet import PureResUNet
from models.degfilm_resunet import DegFiLMResUNet


def benchmark_train_step(model, x, target, n_warmup=5, n_iter=20):
    # 测试一次完整的前向+反向+优化器步骤
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 预热
    for _ in range(n_warmup):
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    times = []
    fwd_times = []
    bwd_times = []
    cpu_percents = []
    
    process = psutil.Process(os.getpid())
    
    for _ in range(n_iter):
        # 记录CPU
        cpu_before = process.cpu_percent(interval=None)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        
        optimizer.zero_grad()
        t1 = time.perf_counter()
        
        pred = model(x)
        t2 = time.perf_counter()
        
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        t3 = time.perf_counter()
        
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        
        # CPU采样（延迟读取）
        cpu_after = process.cpu_percent(interval=None)
        cpu_percents.append(max(cpu_after, 0))
        
        total = (t4 - t0) * 1000
        fwd = (t2 - t1) * 1000
        bwd = (t3 - t2) * 1000
        times.append(total)
        fwd_times.append(fwd)
        bwd_times.append(bwd)
    
    peak_mem = 0
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    return times, fwd_times, bwd_times, cpu_percents, peak_mem


def main():
    print("=" * 60)
    print("A/B 方案训练步骤性能对比 (forward + backward)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print()
    
    shape = (2, 3, 256, 256)  # 模拟 batch_size=2
    x = torch.randn(*shape).to(device)
    target = torch.randn(*shape).to(device)
    
    # A方案
    print("测试 A 方案...")
    model_a = PureResUNet(base_ch=32).to(device).train()
    t_a, fwd_a, bwd_a, cpu_a, mem_a = benchmark_train_step(model_a, x, target)
    del model_a
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # B方案
    print("测试 B 方案...")
    model_b = DegFiLMResUNet(base_ch=32).to(device).train()
    t_b, fwd_b, bwd_b, cpu_b, mem_b = benchmark_train_step(model_b, x, target)
    del model_b
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    import numpy as np
    
    def stats(arr):
        return np.mean(arr), np.std(arr), np.median(arr), np.max(arr)
    
    print(f"\n分辨率: {shape}")
    print("-" * 50)
    
    for label, t, fwd, bwd, cpu, mem in [
        ("A方案", t_a, fwd_a, bwd_a, cpu_a, mem_a),
        ("B方案", t_b, fwd_b, bwd_b, cpu_b, mem_b),
    ]:
        mean, std, med, mx = stats(t)
        fmean, _, _, _ = stats(fwd)
        bmean, _, _, _ = stats(bwd)
        cmean = np.mean([c for c in cpu if c > 0]) if any(c > 0 for c in cpu) else 0
        print(f"  {label}:")
        print(f"    总耗时: {mean:.2f} ± {std:.2f} ms (max {mx:.2f})")
        print(f"    forward: {fmean:.2f} ms, backward: {bmean:.2f} ms")
        print(f"    CPU占用: {cmean:.1f}%")
        if device.type == 'cuda':
            print(f"    峰值显存: {mem:.1f} MB")
    
    overhead = (np.mean(t_b) - np.mean(t_a)) / np.mean(t_a) * 100
    bwd_overhead = (np.mean(bwd_b) - np.mean(bwd_a)) / np.mean(bwd_a) * 100
    print(f"\n  B/A 总开销: {overhead:.1f}%")
    print(f"  B/A backward 开销: {bwd_overhead:.1f}%")
    
    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == '__main__':
    main()
