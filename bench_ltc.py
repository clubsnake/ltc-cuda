"""Microbenchmark for LTC forward/backward comparing TorchScript vs CUDA kernels.

Usage:
    python bench_ltc.py              # Run benchmark
    python bench_ltc.py --check      # Correctness check only
    python bench_ltc.py --profile    # Include kernel launch count
    python bench_ltc.py --deep-profile  # Detailed torch.profiler output
    python bench_ltc.py --save       # Save results to bench_results.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import torch
import torch.nn.functional as F

from ltc import TrueLiquidTimeConstant


def create_ltc(input_dim: int = 128, hidden_dim: int = 64, device: str = "cuda") -> TrueLiquidTimeConstant:
    """Create an LTC module with default training dimensions."""
    ltc = TrueLiquidTimeConstant(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dt=1.0 / 30.0,
        ode_unfolds=6,
        sparsity=0.5,
        sensory_sparsity=0.5,
        input_mapping="affine",
    )
    return ltc.to(device)


def bench_forward(ltc: TrueLiquidTimeConstant, x: torch.Tensor, state: tuple, warmup: int = 10, iters: int = 100) -> float:
    """Benchmark forward pass, returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            ltc(x, state)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            ltc(x, state)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def bench_forward_backward(ltc: TrueLiquidTimeConstant, x: torch.Tensor, state: tuple, warmup: int = 10, iters: int = 100) -> tuple[float, float]:
    """Benchmark forward + backward pass. Returns (forward_ms, total_ms)."""
    target = torch.randn(x.shape[0], x.shape[1], ltc.hidden_dim, device=x.device)

    # Warmup
    for _ in range(warmup):
        out, _ = ltc(x, state)
        loss = F.mse_loss(out, target)
        loss.backward()
        ltc.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    fwd_times = []
    total_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        start.record()
        out, _ = ltc(x, state)
        fwd_end.record()
        loss = F.mse_loss(out, target)
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()

        fwd_times.append(start.elapsed_time(fwd_end))
        total_times.append(start.elapsed_time(bwd_end))
        ltc.zero_grad(set_to_none=True)

    fwd_times.sort()
    total_times.sort()
    mid = len(fwd_times) // 2
    return fwd_times[mid], total_times[mid]


def count_kernel_launches(ltc: TrueLiquidTimeConstant, x: torch.Tensor, state: tuple) -> int:
    """Count CUDA kernel launches in a single forward pass."""
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            ltc(x, state)
            torch.cuda.synchronize()

    # Count CUDA kernel events
    count = 0
    for evt in prof.key_averages():
        if hasattr(evt, 'self_device_time_total') and evt.self_device_time_total > 0:
            count += evt.count
        elif hasattr(evt, 'device_time_total') and evt.device_time_total > 0:
            count += evt.count
    return count


def check_memory(ltc: TrueLiquidTimeConstant, x: torch.Tensor, state: tuple) -> float:
    """Measure peak GPU memory during forward+backward in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    target = torch.randn(x.shape[0], x.shape[1], ltc.hidden_dim, device=x.device)
    out, _ = ltc(x, state)
    loss = F.mse_loss(out, target)
    loss.backward()
    torch.cuda.synchronize()
    ltc.zero_grad(set_to_none=True)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return peak_mb


def check_correctness(device: str = "cuda") -> bool:
    """Compare CUDA kernel output against TorchScript reference (forward + backward)."""
    try:
        from ltc_cuda import _HAS_CUDA_KERNELS
        if not _HAS_CUDA_KERNELS:
            print("CUDA kernels not available, skipping correctness check")
            return True
    except ImportError:
        print("ltc_cuda module not found, skipping correctness check")
        return True

    all_pass = True

    # --- Forward correctness ---
    torch.manual_seed(42)
    ltc = create_ltc(input_dim=32, hidden_dim=16, device=device)
    x = torch.randn(4, 8, 32, device=device)
    state = (torch.randn(4, 16, device=device), torch.randn(4, 16, device=device))

    # TorchScript reference
    ltc._force_torchscript = True
    with torch.no_grad():
        ref_out, ref_state = ltc(x, state)

    # CUDA kernel
    ltc._force_torchscript = False
    ltc._cached_transforms = None
    with torch.no_grad():
        cuda_out, cuda_state = ltc(x, state)

    # Compare forward (relaxed tolerance for --use_fast_math)
    fwd_match = torch.allclose(ref_out, cuda_out, rtol=1e-2, atol=1e-3)
    state_match = torch.allclose(ref_state[0], cuda_state[0], rtol=1e-2, atol=1e-3)

    max_diff_out = (ref_out - cuda_out).abs().max().item()
    max_diff_state = (ref_state[0] - cuda_state[0]).abs().max().item()
    if fwd_match and state_match:
        print(f"Forward correctness PASSED (output max diff: {max_diff_out:.2e}, state: {max_diff_state:.2e})")
    else:
        print(f"Forward correctness FAILED: output diff={max_diff_out:.2e}, state diff={max_diff_state:.2e}")
        all_pass = False

    # --- Backward correctness ---
    torch.manual_seed(42)
    ltc2 = create_ltc(input_dim=32, hidden_dim=16, device=device)
    x2 = torch.randn(2, 4, 32, device=device)

    # TorchScript grads
    ltc2._force_torchscript = True
    ltc2.zero_grad()
    out_ts, _ = ltc2(x2, None)
    out_ts.sum().backward()
    ts_grads = {n: p.grad.clone() for n, p in ltc2.named_parameters() if p.grad is not None}

    # CUDA grads
    ltc2._force_torchscript = False
    ltc2._cached_transforms = None
    ltc2.zero_grad()
    out_cuda, _ = ltc2(x2, None)
    out_cuda.sum().backward()

    max_grad_err = 0.0
    worst_param = ""
    for name, param in ltc2.named_parameters():
        if name in ts_grads and param.grad is not None:
            err = (param.grad - ts_grads[name]).abs().max().item()
            if err > max_grad_err:
                max_grad_err = err
                worst_param = name

    if max_grad_err < 0.05:
        print(f"Backward correctness PASSED (worst grad diff: {max_grad_err:.2e} in {worst_param})")
    else:
        print(f"Backward correctness FAILED: worst grad diff={max_grad_err:.2e} in {worst_param}")
        for name, param in ltc2.named_parameters():
            if name in ts_grads and param.grad is not None:
                err = (param.grad - ts_grads[name]).abs().max().item()
                if err > 0.01:
                    print(f"  {name}: {err:.2e}")
        all_pass = False

    return all_pass


def gpu_status():
    """Print current GPU clocks, temp, power."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        if len(parts) >= 5:
            print(f"  GPU: {parts[0]}MHz core, {parts[1]}MHz mem, {parts[2]}C, {parts[3]}W, {parts[4]}")
    except Exception:
        pass


def bench_forward_burst(ltc, x, state, warmup=5, iters=10):
    """Benchmark with short burst to avoid thermal throttling."""
    for _ in range(warmup):
        with torch.no_grad():
            ltc(x, state)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            ltc(x, state)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench_fwd_bwd_burst(ltc, x, state, warmup=5, iters=10):
    """Benchmark fwd+bwd with short burst to avoid throttling."""
    target = torch.randn(x.shape[0], x.shape[1], ltc.hidden_dim, device=x.device)
    for _ in range(warmup):
        out, _ = ltc(x, state)
        loss = F.mse_loss(out, target)
        loss.backward()
        ltc.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    fwd_times = []
    total_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        start.record()
        out, _ = ltc(x, state)
        fwd_end.record()
        loss = F.mse_loss(out, target)
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()

        fwd_times.append(start.elapsed_time(fwd_end))
        total_times.append(start.elapsed_time(bwd_end))
        ltc.zero_grad(set_to_none=True)
    fwd_times.sort()
    total_times.sort()
    mid = len(fwd_times) // 2
    return fwd_times[mid], total_times[mid]


def profile_detailed(ltc, x, state):
    """Detailed profiling with torch.profiler -- shows per-kernel breakdown."""
    target = torch.randn(x.shape[0], x.shape[1], ltc.hidden_dim, device=x.device)
    mem_before = torch.cuda.memory_stats()

    # Warmup
    for _ in range(3):
        out, _ = ltc(x, state)
        loss = F.mse_loss(out, target)
        loss.backward()
        ltc.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        out, _ = ltc(x, state)
        loss = F.mse_loss(out, target)
        loss.backward()
        torch.cuda.synchronize()
        ltc.zero_grad(set_to_none=True)

    print("\n--- Profiler: Top 30 CUDA operations ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Count operations
    kernel_launches = 0
    memsets = 0
    memcpys = 0
    sync_points = 0
    total_cuda_us = 0
    buckets: collections.defaultdict[str, list[float | int]] = collections.defaultdict(lambda: [0, 0.0])

    def bucket_name(name: str) -> str:
        n = name.lower()
        if "ltc_ode_step_bwd_fused_kernel" in n:
            return "ode_bwd_fused"
        if "ltc_ode_step_bwd_state_kernel" in n:
            return "ode_bwd_state"
        if "ltc_ode_step_bwd_params_kernel" in n:
            return "ode_bwd_params"
        if "ltc_sensory_bwd_kernel" in n:
            return "sensory_bwd"
        if "ltc_ode_step_fwd_kernel" in n:
            return "ode_fwd"
        if "ltc_sensory_fwd_kernel" in n:
            return "sensory_fwd"
        if "memset" in n:
            return "memset"
        if "memcpy" in n:
            return "memcpy"
        if "copy_" in n:
            return "copy"
        if "transpose" in n or "contiguous" in n:
            return "layout"
        return "other"

    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            total_cuda_us += evt.self_device_time_total
            name_lower = evt.key.lower()
            if 'memset' in name_lower:
                memsets += evt.count
            elif 'memcpy' in name_lower:
                memcpys += evt.count
            else:
                kernel_launches += evt.count
            b = bucket_name(evt.key)
            buckets[b][0] += evt.count
            buckets[b][1] += evt.self_device_time_total
        if "synchronize" in evt.key.lower():
            sync_points += evt.count

    print(f"\nOperation counts: {kernel_launches} kernel launches, {memsets} memsets, {memcpys} memcpys")
    print(f"CPU sync points: {sync_points}")
    print(f"Total CUDA time: {total_cuda_us/1000:.2f} ms")
    if buckets:
        print("\nKernel buckets (self CUDA):")
        for name, (count, total_us) in sorted(buckets.items(), key=lambda kv: kv[1][1], reverse=True):
            if total_us <= 0:
                continue
            print(f"  {name:16s} {int(count):4d} launches  {total_us/1000:7.3f} ms")

    mem_after = torch.cuda.memory_stats()
    alloc_delta = mem_after.get("allocation.all.current", 0) - mem_before.get("allocation.all.current", 0)
    seg_delta = mem_after.get("segment.all.current", 0) - mem_before.get("segment.all.current", 0)
    print("\nAllocator stats delta:")
    print(f"  allocation.all.current: {alloc_delta:+d}")
    print(f"  segment.all.current:    {seg_delta:+d}")
    print(f"  active_bytes.all.peak:  {mem_after.get('active_bytes.all.peak', 0)/(1024*1024):.2f} MB")
    return prof


def main():
    parser = argparse.ArgumentParser(description="LTC Benchmark")
    parser.add_argument("--check", action="store_true", help="Run correctness check only")
    parser.add_argument("--profile", action="store_true", help="Count kernel launches")
    parser.add_argument("--deep-profile", action="store_true", help="Detailed torch.profiler output")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--burst", type=int, default=10, help="Burst iterations (throttle-safe)")
    parser.add_argument("--batch", type=int, default=24, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length")
    parser.add_argument("--input-dim", type=int, default=128, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: B={args.batch}, T={args.seq_len}, D={args.input_dim}, H={args.hidden_dim}")
    print()

    if args.check:
        check_correctness(device)
        return

    torch.manual_seed(42)
    ltc = create_ltc(args.input_dim, args.hidden_dim, device)
    x = torch.randn(args.batch, args.seq_len, args.input_dim, device=device)
    state = (
        torch.zeros(args.batch, args.hidden_dim, device=device),
        torch.zeros(args.batch, args.hidden_dim, device=device),
    )

    param_count = sum(p.numel() for p in ltc.parameters())
    print(f"LTC parameters: {param_count:,}")
    gpu_status()

    # --- TorchScript Baseline (burst mode to avoid throttling) ---
    print("\n--- TorchScript Baseline (burst) ---")
    ltc._force_torchscript = True

    fwd_ms = bench_forward_burst(ltc, x, state, iters=args.burst)
    print(f"Forward only:      {fwd_ms:.2f} ms (median of {args.burst})")

    fwd_fb_ms, total_ms = bench_fwd_bwd_burst(ltc, x, state, iters=args.burst)
    bwd_ms = total_ms - fwd_fb_ms
    print(f"Forward (in f+b):  {fwd_fb_ms:.2f} ms")
    print(f"Backward:          {bwd_ms:.2f} ms")
    print(f"Total (fwd+bwd):   {total_ms:.2f} ms")

    peak_mb = check_memory(ltc, x, state)
    print(f"Peak GPU memory:   {peak_mb:.1f} MB")
    gpu_status()

    if args.profile:
        kernel_count = count_kernel_launches(ltc, x, state)
        print(f"Kernel launches:   {kernel_count}")

    ltc._force_torchscript = False

    # --- CUDA Kernel Mode ---
    try:
        from ltc_cuda import _HAS_CUDA_KERNELS
        if _HAS_CUDA_KERNELS:
            print("\n--- CUDA Kernel Mode (burst) ---")
            ltc._cached_transforms = None

            fwd_cuda = bench_forward_burst(ltc, x, state, iters=args.burst)
            fwd_fb_cuda, total_cuda = bench_fwd_bwd_burst(ltc, x, state, iters=args.burst)
            bwd_cuda = total_cuda - fwd_fb_cuda
            print(f"Forward only:      {fwd_cuda:.2f} ms ({fwd_ms/fwd_cuda:.1f}x speedup)")
            print(f"Forward (in f+b):  {fwd_fb_cuda:.2f} ms")
            print(f"Backward:          {bwd_cuda:.2f} ms ({bwd_ms/bwd_cuda:.1f}x speedup)")
            print(f"Total (fwd+bwd):   {total_cuda:.2f} ms ({total_ms/total_cuda:.1f}x speedup)")

            peak_cuda_mb = check_memory(ltc, x, state)
            print(f"Peak GPU memory:   {peak_cuda_mb:.1f} MB ({peak_mb/peak_cuda_mb:.1f}x less)")
            gpu_status()

            if args.profile:
                k_cuda = count_kernel_launches(ltc, x, state)
                print(f"Kernel launches:   {k_cuda}")

            if args.deep_profile:
                print("\n--- Deep Profile: TorchScript ---")
                ltc._force_torchscript = True
                ltc._cached_transforms = None
                profile_detailed(ltc, x, state)

                print("\n--- Deep Profile: CUDA Kernels ---")
                ltc._force_torchscript = False
                ltc._cached_transforms = None
                profile_detailed(ltc, x, state)
    except ImportError:
        pass

    # Correctness
    print()
    check_correctness(device)

    if args.save:
        results = {
            "gpu": torch.cuda.get_device_name(),
            "batch": args.batch,
            "seq_len": args.seq_len,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "params": param_count,
            "ts_forward_ms": round(fwd_ms, 2),
            "ts_backward_ms": round(bwd_ms, 2),
            "ts_total_ms": round(total_ms, 2),
            "ts_peak_mb": round(peak_mb, 1),
        }
        if _HAS_CUDA_KERNELS:
            results.update({
                "cuda_forward_ms": round(fwd_cuda, 2),
                "cuda_backward_ms": round(bwd_cuda, 2),
                "cuda_total_ms": round(total_cuda, 2),
                "cuda_peak_mb": round(peak_cuda_mb, 1),
                "fwd_speedup": round(fwd_ms / fwd_cuda, 2),
                "bwd_speedup": round(bwd_ms / bwd_cuda, 2),
                "total_speedup": round(total_ms / total_cuda, 2),
            })
        out_path = os.path.join(os.path.dirname(__file__), "bench_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
