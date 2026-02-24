"""Microbenchmark for LTC forward/backward comparing TorchScript vs CUDA kernels.

Usage:
    python bench_ltc.py              # Run benchmark
    python bench_ltc.py --check      # Correctness check only
    python bench_ltc.py --profile    # Include kernel launch count
    python bench_ltc.py --save       # Save results to bench_results.json
"""

from __future__ import annotations

import argparse
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
    """Compare CUDA kernel output against TorchScript reference."""
    try:
        from ltc_cuda import LTCCudaCell, _HAS_CUDA_KERNELS
        if not _HAS_CUDA_KERNELS:
            print("CUDA kernels not available, skipping correctness check")
            return True
    except ImportError:
        print("ltc_cuda module not found, skipping correctness check")
        return True

    torch.manual_seed(42)
    ltc = create_ltc(input_dim=32, hidden_dim=16, device=device)
    x = torch.randn(4, 8, 32, device=device)
    state = (torch.randn(4, 16, device=device), torch.randn(4, 16, device=device))

    # TorchScript reference
    with torch.no_grad():
        ref_out, ref_state = ltc(x, state)

    # CUDA kernel
    ltc._force_cuda_kernels = True
    with torch.no_grad():
        cuda_out, cuda_state = ltc(x, state)
    ltc._force_cuda_kernels = False

    # Compare
    fwd_match = torch.allclose(ref_out, cuda_out, rtol=1e-4, atol=1e-5)
    state_match = torch.allclose(ref_state[0], cuda_state[0], rtol=1e-4, atol=1e-5)

    if fwd_match and state_match:
        max_diff = (ref_out - cuda_out).abs().max().item()
        print(f"Correctness check PASSED (max diff: {max_diff:.2e})")
        return True
    else:
        max_diff_out = (ref_out - cuda_out).abs().max().item()
        max_diff_state = (ref_state[0] - cuda_state[0]).abs().max().item()
        print(f"Correctness check FAILED: output diff={max_diff_out:.2e}, state diff={max_diff_state:.2e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LTC Benchmark")
    parser.add_argument("--check", action="store_true", help="Run correctness check only")
    parser.add_argument("--profile", action="store_true", help="Count kernel launches")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
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

    # --- TorchScript Baseline ---
    print("\n--- TorchScript Baseline ---")
    ltc._force_torchscript = True

    fwd_ms = bench_forward(ltc, x, state, iters=args.iters)
    print(f"Forward only:      {fwd_ms:.2f} ms (median of {args.iters})")

    fwd_fb_ms, total_ms = bench_forward_backward(ltc, x, state, iters=args.iters)
    bwd_ms = total_ms - fwd_fb_ms
    print(f"Forward (in f+b):  {fwd_fb_ms:.2f} ms")
    print(f"Backward:          {bwd_ms:.2f} ms")
    print(f"Total (fwd+bwd):   {total_ms:.2f} ms")

    peak_mb = check_memory(ltc, x, state)
    print(f"Peak GPU memory:   {peak_mb:.1f} MB")

    if args.profile:
        kernel_count = count_kernel_launches(ltc, x, state)
        print(f"Kernel launches:   {kernel_count}")

    ltc._force_torchscript = False

    # --- CUDA Kernel Mode ---
    try:
        from ltc_cuda import _HAS_CUDA_KERNELS
        if _HAS_CUDA_KERNELS:
            print("\n--- CUDA Kernel Mode ---")
            # Clear cached transforms to ensure CUDA path is taken
            ltc._cached_transforms = None

            fwd_cuda = bench_forward(ltc, x, state, iters=args.iters)
            fwd_fb_cuda, total_cuda = bench_forward_backward(ltc, x, state, iters=args.iters)
            bwd_cuda = total_cuda - fwd_fb_cuda
            print(f"Forward only:      {fwd_cuda:.2f} ms ({fwd_ms/fwd_cuda:.2f}x speedup)")
            print(f"Forward (in f+b):  {fwd_fb_cuda:.2f} ms")
            print(f"Backward:          {bwd_cuda:.2f} ms ({bwd_ms/bwd_cuda:.2f}x speedup)")
            print(f"Total (fwd+bwd):   {total_cuda:.2f} ms ({total_ms/total_cuda:.2f}x speedup)")

            peak_cuda_mb = check_memory(ltc, x, state)
            print(f"Peak GPU memory:   {peak_cuda_mb:.1f} MB")

            if args.profile:
                k_cuda = count_kernel_launches(ltc, x, state)
                print(f"Kernel launches:   {k_cuda}")
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
            "forward_ms": round(fwd_ms, 2),
            "backward_ms": round(bwd_ms, 2),
            "total_ms": round(total_ms, 2),
            "peak_mb": round(peak_mb, 1),
        }
        if args.profile:
            results["kernel_count"] = kernel_count
        out_path = os.path.join(os.path.dirname(__file__), "bench_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
