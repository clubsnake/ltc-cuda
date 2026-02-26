# ltc-cuda

Drop-in CUDA acceleration for [Liquid Time-Constant (LTC)](https://arxiv.org/abs/2006.04439) neural networks. **8x faster forward+backward, 35x less memory** compared to TorchScript.

## Performance

Measured on GTX 1080 (sm_61), B=24, T=8, D=128, H=64, 6 ODE unfolds:

| Metric | TorchScript | CUDA Kernels | Speedup |
|--------|-------------|--------------|---------|
| Forward | 14.7 ms | 1.7 ms | **8.6x** |
| Backward | 74 ms | 11 ms | **6.8x** |
| **Total** | **89 ms** | **12.7 ms** | **7.0x** |
| Peak Memory* | 71 MB | 2 MB | **35x** |
| Kernel Launches | ~2000 | 16 | **125x** |

*\*Peak memory measured via `torch.cuda.max_memory_allocated()` during forward+backward. TorchScript stores intermediate activations for autograd; CUDA kernels recompute them in the backward pass. Reproduce with `python bench_ltc.py --save`.*

## Quick Start

```bash
git clone https://github.com/clubsnake/ltc-cuda.git
cd ltc-cuda
pip install -r requirements.txt
python bench_ltc.py
```

CUDA kernels are **JIT-compiled on first use** (takes ~30s). After that, the compiled `.pyd`/`.so` is cached in `_cuda_build/` and reused automatically.

## Requirements

- Python 3.9+
- PyTorch >= 2.0 with CUDA
- NVIDIA GPU (tested on sm_61/Pascal, should work on sm_70+)
- **Windows**: [MSVC Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (C++ Desktop workload)
- **Linux**: GCC with C++17 support
- CUDA Toolkit matching your PyTorch CUDA version (check with `python -c "import torch; print(torch.version.cuda)"`)

## Usage

```python
from ltc import TrueLiquidTimeConstant

ltc = TrueLiquidTimeConstant(
    input_dim=128,         # your encoder output dimension
    hidden_dim=64,         # number of LTC neurons
    dt=1/30,               # integration timestep (seconds)
    ode_unfolds=6,         # ODE sub-steps per timestep
    sparsity=0.5,          # recurrent connection density
    sensory_sparsity=0.5,  # input connection density
).cuda()

# x: (batch, seq_len, input_dim)
x = torch.randn(16, 32, 128, device="cuda")
output, (h, c) = ltc(x)  # output: (16, 32, 64)
```

CUDA kernels activate automatically when:
- Input is on CUDA and dtype is float32
- CUDA toolkit + C++ compiler are available for JIT compilation

Falls back to TorchScript transparently if CUDA compilation fails.

## Benchmark

```bash
# Default benchmark (B=24, T=8, D=128, H=64)
python bench_ltc.py

# Custom dimensions
python bench_ltc.py --batch 32 --seq-len 16 --hidden-dim 128

# Correctness check only (CUDA vs TorchScript)
python bench_ltc.py --check

# Count kernel launches
python bench_ltc.py --profile

# Save results to bench_results.json
python bench_ltc.py --save
```

## Tests

```bash
# Correctness: CUDA output vs TorchScript reference
python -m pytest test_ltc_cuda.py -v

# Verification: gradient accuracy, determinism, spec compliance
python -m pytest test_ltc_verification.py -v

# Run all tests
python -m pytest -v
```

## Architecture

The standard LTC forward pass requires **~2000 CUDA kernel launches** per step (from TorchScript decomposing each op). We replace this with **5 hand-written CUDA kernels** wrapped in C++ time-loop functions:

| Kernel | Registers | What it does |
|--------|-----------|-------------|
| `sensory_fwd` | 48 | Input conductance: `w * sigmoid(sigma * (x - mu))`, reduces (B,D,H) to (B,H) |
| `ode_fwd` | 40 | One ODE unfold step: recurrent conductance + numerator/denominator update |
| `sensory_bwd` | 26 | Backward pass for sensory synapses (per-batch gradient accumulation) |
| `ode_bwd_fused` | ~56 | Fused state + parameter gradients through ODE unfolds |

**Key design decisions:**
- **Per-batch gradient accumulation**: Each batch element accumulates into its own gradient buffer, then a final reduction sums across batches. Eliminates atomicAdd contention that caused 6x slowdown in the backward pass.
- **Fused backward kernel**: State and parameter gradients are computed in a single kernel launch per ODE unfold, reducing launch overhead by ~40%.
- **Pre-transposed matrices**: Forward pass returns transposed weight matrices that backward reuses directly, avoiding 8 transpose operations.
- **C++ T-loop wrappers**: `ltc_full_forward()` and `ltc_full_backward()` loop over timesteps in C++, eliminating Python-level dispatch for each timestep
- **fp32 only**: LTC's 6 ODE unfolds accumulate numerical error; fp16 diverges. The kernels use `--use_fast_math` for speed but maintain fp32 precision throughout
- **Stride-aware sensory kernel**: Accepts non-contiguous input tensors, avoiding `.contiguous()` copies

## Numerical Stability

- All intermediate values computed in fp32 (no fp16 path)
- Epsilon (1e-8) in denominator prevents division by zero
- Softplus ensures positivity of weights, membrane capacitance, and leak conductance
- `--use_fast_math` enables FMA contraction (slightly different accumulation order vs TorchScript, max diff ~1e-3)
- Gradient verification: CUDA gradients match TorchScript within rtol=5e-2, atol=1e-2

## How It Works

LTC neurons follow the ODE:

```
cm * dv/dt = -gleak * (v - vleak) - sum_j [ g_ij * (v - erev_ij) ]
```

where `g_ij = softplus(w_ij) * sigmoid(sigma_ij * (v_j - mu_ij))` is the per-synapse conductance gate. This is solved via implicit Euler with `ode_unfolds` sub-steps per timestep.

The TorchScript implementation decomposes this into hundreds of small PyTorch ops (each launching a CUDA kernel). Our kernels fuse the entire ODE step into a single launch, with the time loop running in C++.

## Compatibility

| | Status |
|--|--------|
| **Windows** | Tested (MSVC 2022 + CUDA 12.6) |
| **Linux** | Should work (GCC + CUDA) |
| **sm_61** (Pascal) | Tested (GTX 1080) |
| **sm_70+** (Volta/Ampere/Hopper) | Should work (not yet tested) |
| **PyTorch 2.0+** | Required for `torch.utils.cpp_extension` |

## API Reference

### `TrueLiquidTimeConstant(input_dim, hidden_dim, dt, ode_unfolds, sparsity, sensory_sparsity, input_mapping, epsilon)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | required | Input feature dimension |
| `hidden_dim` | int | 64 | Number of LTC neurons |
| `dt` | float | 1/30 | Integration timestep in seconds |
| `ode_unfolds` | int | 6 | ODE sub-steps per timestep (more = more accurate, slower) |
| `sparsity` | float | 0.5 | Fraction of recurrent connections (0=none, 1=fully connected) |
| `sensory_sparsity` | float | 0.5 | Fraction of input-to-state connections |
| `input_mapping` | str | "affine" | Input preprocessing: "affine", "linear", or "none" |
| `epsilon` | float | 1e-8 | Numerical stability constant |

**Forward signature**: `forward(x_seq, h0=None) -> (outputs, (h, c))`
- `x_seq`: `(B, T, input_dim)` input sequence
- `h0`: optional `(h, c)` initial state tuple (LSTM-compatible API)
- Returns: `outputs (B, T, hidden_dim)`, final state `(h, h)` (c mirrors h)

## Files

| File | Description |
|------|-------------|
| `ltc.py` | LTC module with TorchScript fallback + CUDA dispatch |
| `ltc_cuda.py` | CUDA kernel JIT loader + autograd Function wrapper |
| `csrc/ltc_kernels.cu` | CUDA kernels with fused backward + per-batch accumulators (~1300 lines) |
| `recurrent.py` | LSTM wrapper for benchmark comparison |
| `bench_ltc.py` | Benchmark script |
| `test_ltc_cuda.py` | CUDA vs TorchScript correctness tests |
| `test_ltc_verification.py` | LTC spec compliance + gradient tests |

## Citation

If you use this code, please cite the original LTC paper:

```bibtex
@article{hasani2021liquid,
  title={Liquid Time-constant Networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and King, Gerald and Rus, Daniela},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## License

MIT
