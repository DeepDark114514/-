import sys
sys.path.insert(0, r'D:\biyesheji\DLC\NEW')

import torch
import time
import gc
from models.pure_resunet import PureResUNet
from models.degfilm_resunet import DegFiLMResUNet


def benchmark(model, x, n_warmup=10, n_iter=50):
    # 测试前向传播速度和显存占用
    model.eval()
    device = next(model.parameters()).device
    
    # 清理缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    
    peak_mem = 0
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return times, peak_mem


def main():
    print("=" * 60)
    print("A/B 方案前向传播性能对比")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 不同分辨率测试
    resolutions = [
        (1, 3, 256, 256),
        (1, 3, 512, 512),
        (1, 3, 720, 1280),
    ]
    
    for shape in resolutions:
        print(f"\n分辨率: {shape}")
        print("-" * 40)
        x = torch.randn(*shape).to(device)
        
        # A方案
        model_a = PureResUNet(base_ch=32).to(device).eval()
        times_a, mem_a = benchmark(model_a, x)
        del model_a
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # B方案
        model_b = DegFiLMResUNet(base_ch=32).to(device).eval()
        times_b, mem_b = benchmark(model_b, x)
        del model_b
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        import numpy as np
        mean_a = np.mean(times_a)
        std_a = np.std(times_a)
        mean_b = np.mean(times_b)
        std_b = np.std(times_b)
        
        print(f"  A方案: {mean_a:.2f} ± {std_a:.2f} ms", end="")
        if device.type == 'cuda':
            print(f"  峰值显存: {mem_a:.1f} MB")
        else:
            print()
        
        print(f"  B方案: {mean_b:.2f} ± {std_b:.2f} ms", end="")
        if device.type == 'cuda':
            print(f"  峰值显存: {mem_b:.1f} MB")
        else:
            print()
        
        overhead = (mean_b - mean_a) / mean_a * 100
        print(f"  B/A 开销: {overhead:.1f}%")
        
        if device.type == 'cuda':
            mem_overhead = (mem_b - mem_a) / mem_a * 100
            print(f"  显存开销: {mem_overhead:.1f}%")
    
    print("\n" + "=" * 60)
    
    # 额外测试：连续 forward 1000 次，检测是否越来越慢（泄漏/累积问题）
    print("\n[泄漏测试] 连续 forward 1000 次...")
    x = torch.randn(1, 3, 256, 256).to(device)
    model_b = DegFiLMResUNet(base_ch=32).to(device).eval()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(1000):
            _ = model_b(x)
            if i % 200 == 199:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                mem = torch.cuda.memory_allocated() / 1024**2 if device.type == 'cuda' else 0
                print(f"  iter {i+1}: {(t1-t0)*1000/(i+1):.3f} ms/iter, 显存: {mem:.1f} MB")
    
    del model_b
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n测试完成")


if __name__ == '__main__':
    main()
