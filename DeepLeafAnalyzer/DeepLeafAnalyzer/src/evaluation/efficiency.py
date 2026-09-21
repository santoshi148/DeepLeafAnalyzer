import time
import torch

def count_parameters(model): return sum(p.numel() for p in model.parameters())

def latency_ms(model, sample, device, warmup=3, runs=10):
    model.eval(); sample=sample.to(device)
    with torch.no_grad():
        for _ in range(warmup): model(sample)
        if device.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter()
        for _ in range(runs): model(sample)
        if device.type=='cuda': torch.cuda.synchronize()
    return (time.perf_counter()-t0)*1000/runs
