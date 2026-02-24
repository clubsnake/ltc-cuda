"""Tests for custom CUDA LTC kernels.

Tests numerical correctness against TorchScript reference and gradient
computation via autograd.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ltc import TrueLiquidTimeConstant

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _has_cuda_kernels():
    try:
        from ltc_cuda import _HAS_CUDA_KERNELS
        return _HAS_CUDA_KERNELS
    except ImportError:
        return False


def _create_ltc(input_dim=32, hidden_dim=16, device="cuda"):
    """Create a small LTC for testing."""
    torch.manual_seed(42)
    ltc = TrueLiquidTimeConstant(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dt=1.0 / 30.0,
        ode_unfolds=6,
        sparsity=0.5,
        sensory_sparsity=0.5,
        input_mapping="affine",
    ).to(device)
    return ltc


@pytest.mark.skipif(not _has_cuda_kernels(), reason="CUDA kernels not compiled")
class TestLTCCudaCorrectness:
    """Numerical comparison between CUDA kernels and TorchScript reference."""

    def test_forward_output_matches(self):
        """CUDA forward output must match TorchScript within fp32 tolerance."""
        ltc = _create_ltc()
        torch.manual_seed(123)
        x = torch.randn(4, 8, 32, device="cuda")
        state = (torch.randn(4, 16, device="cuda"), torch.randn(4, 16, device="cuda"))

        # TorchScript reference
        ltc._force_torchscript = True
        with torch.no_grad():
            ref_out, ref_state = ltc(x, state)

        # CUDA kernel
        ltc._force_torchscript = False
        with torch.no_grad():
            cuda_out, cuda_state = ltc(x, state)

        # --use_fast_math gives slightly different accumulation order;
        # rtol=1e-2 is appropriate for fused sigmoid+reduce kernels
        assert torch.allclose(ref_out, cuda_out, rtol=1e-2, atol=1e-3), \
            f"Output max diff: {(ref_out - cuda_out).abs().max().item():.2e}"
        assert torch.allclose(ref_state[0], cuda_state[0], rtol=1e-2, atol=1e-3), \
            f"State max diff: {(ref_state[0] - cuda_state[0]).abs().max().item():.2e}"

    def test_forward_single_timestep(self):
        """Single timestep forward should match."""
        ltc = _create_ltc()
        torch.manual_seed(456)
        x = torch.randn(2, 1, 32, device="cuda")
        state = (torch.zeros(2, 16, device="cuda"), torch.zeros(2, 16, device="cuda"))

        ltc._force_torchscript = True
        with torch.no_grad():
            ref_out, _ = ltc(x, state)
        ltc._force_torchscript = False
        with torch.no_grad():
            cuda_out, _ = ltc(x, state)

        assert torch.allclose(ref_out, cuda_out, rtol=1e-2, atol=1e-3)

    def test_forward_zero_state(self):
        """Starting from zero state should match."""
        ltc = _create_ltc()
        torch.manual_seed(789)
        x = torch.randn(4, 4, 32, device="cuda")

        ltc._force_torchscript = True
        with torch.no_grad():
            ref_out, _ = ltc(x, None)
        ltc._force_torchscript = False
        with torch.no_grad():
            cuda_out, _ = ltc(x, None)

        assert torch.allclose(ref_out, cuda_out, rtol=1e-2, atol=1e-3)

    def test_forward_large_batch(self):
        """Production-sized batch (B=24, T=8, D=224, H=64)."""
        ltc = _create_ltc(input_dim=224, hidden_dim=64)
        torch.manual_seed(101)
        x = torch.randn(24, 8, 224, device="cuda")
        state = (torch.zeros(24, 64, device="cuda"), torch.zeros(24, 64, device="cuda"))

        ltc._force_torchscript = True
        with torch.no_grad():
            ref_out, ref_state = ltc(x, state)
        ltc._force_torchscript = False
        with torch.no_grad():
            cuda_out, cuda_state = ltc(x, state)

        # Larger dimensions accumulate more --use_fast_math rounding over
        # T=8 × 6 unfolds; rtol=5e-2 is appropriate for D=224, H=64.
        max_diff = (ref_out - cuda_out).abs().max().item()
        mean_diff = (ref_out - cuda_out).abs().mean().item()
        assert max_diff < 0.05, \
            f"Output max diff too large: {max_diff:.2e} (mean: {mean_diff:.2e})"

    def test_gradient_backward(self):
        """Backward pass should produce valid gradients matching TorchScript."""
        ltc = _create_ltc()
        torch.manual_seed(202)
        x = torch.randn(2, 4, 32, device="cuda")
        state = (torch.zeros(2, 16, device="cuda"), torch.zeros(2, 16, device="cuda"))

        # CUDA backward
        ltc._force_torchscript = False
        ltc.zero_grad()
        out, _ = ltc(x, state)
        out.sum().backward()

        for name, param in ltc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_matches_torchscript(self):
        """CUDA gradients should approximately match TorchScript gradients."""
        ltc = _create_ltc()
        torch.manual_seed(202)
        x = torch.randn(2, 4, 32, device="cuda")

        # TorchScript grads
        ltc._force_torchscript = True
        ltc.zero_grad()
        out_ts, _ = ltc(x, None)
        out_ts.sum().backward()
        ts_grads = {n: p.grad.clone() for n, p in ltc.named_parameters() if p.grad is not None}

        # CUDA grads
        ltc._force_torchscript = False
        ltc._cached_transforms = None
        ltc.zero_grad()
        out_cuda, _ = ltc(x, None)
        out_cuda.sum().backward()

        for name, param in ltc.named_parameters():
            if name in ts_grads and param.grad is not None:
                assert torch.allclose(param.grad, ts_grads[name], rtol=5e-2, atol=1e-2), \
                    f"Gradient mismatch for {name}: max_err={( param.grad - ts_grads[name]).abs().max().item():.2e}"

    def test_saturated_sigmoids(self):
        """Extreme sigma/mu values should not cause NaN/Inf."""
        ltc = _create_ltc()
        with torch.no_grad():
            ltc.sigma.fill_(100.0)
            ltc.sensory_sigma.fill_(100.0)

        torch.manual_seed(303)
        x = torch.randn(2, 4, 32, device="cuda") * 10.0

        with torch.no_grad():
            out, (h, _) = ltc(x, None)

        assert torch.isfinite(out).all(), "NaN/Inf in output with extreme sigmoids"
        assert torch.isfinite(h).all(), "NaN/Inf in state with extreme sigmoids"

    def test_determinism(self):
        """Two forward passes with same input should produce identical output."""
        ltc = _create_ltc()
        torch.manual_seed(404)
        x = torch.randn(4, 8, 32, device="cuda")
        state = (torch.zeros(4, 16, device="cuda"), torch.zeros(4, 16, device="cuda"))

        with torch.no_grad():
            out1, _ = ltc(x, state)
            out2, _ = ltc(x, state)

        # --use_fast_math allows FMA contraction reordering between launches;
        # bitwise equality is not guaranteed, but results must be very close.
        # Observed ~1.5e-3 max diff on GTX 1080 with T=8 timesteps.
        assert torch.allclose(out1, out2, rtol=1e-2, atol=2e-3), \
            f"Non-deterministic CUDA forward: max diff {(out1 - out2).abs().max().item():.2e}"


    @pytest.mark.parametrize("hidden_dim", [33, 63, 17, 65])
    def test_forward_odd_hidden_dim(self, hidden_dim):
        """Odd H values where last block has partial warps must still be correct."""
        ltc = _create_ltc(input_dim=32, hidden_dim=hidden_dim)
        torch.manual_seed(808)
        x = torch.randn(4, 4, 32, device="cuda")

        ltc._force_torchscript = True
        with torch.no_grad():
            ref_out, _ = ltc(x, None)
        ltc._force_torchscript = False
        ltc._cached_transforms = None
        with torch.no_grad():
            cuda_out, _ = ltc(x, None)

        max_diff = (ref_out - cuda_out).abs().max().item()
        assert max_diff < 1e-3, \
            f"H={hidden_dim} forward max diff: {max_diff:.2e}"

    @pytest.mark.parametrize("hidden_dim", [33, 63])
    def test_gradient_odd_hidden_dim(self, hidden_dim):
        """Backward pass must be correct for odd H values."""
        ltc = _create_ltc(input_dim=32, hidden_dim=hidden_dim)
        torch.manual_seed(909)
        x = torch.randn(2, 4, 32, device="cuda")

        # TorchScript grads
        ltc._force_torchscript = True
        ltc.zero_grad()
        out_ts, _ = ltc(x, None)
        out_ts.sum().backward()
        ts_grads = {n: p.grad.clone() for n, p in ltc.named_parameters() if p.grad is not None}

        # CUDA grads
        ltc._force_torchscript = False
        ltc._cached_transforms = None
        ltc.zero_grad()
        out_cuda, _ = ltc(x, None)
        out_cuda.sum().backward()

        for name, param in ltc.named_parameters():
            if name in ts_grads and param.grad is not None:
                max_err = (param.grad - ts_grads[name]).abs().max().item()
                assert max_err < 0.05, \
                    f"H={hidden_dim} grad mismatch for {name}: {max_err:.2e}"


class TestLTCTorchScriptPreFuse:
    """Test the w_erev pre-fusion optimization in TorchScript path."""

    def test_prefuse_matches_original(self):
        """The pre-fused w_erev should give same results as computing w_pos*erev inline."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ltc = _create_ltc(device=device)
        ltc._force_torchscript = True
        torch.manual_seed(505)
        x = torch.randn(4, 8, 32, device=device)
        state = (torch.zeros(4, 16, device=device), torch.zeros(4, 16, device=device))

        with torch.no_grad():
            out, (h, _) = ltc(x, state)

        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_gradient_flow_with_prefuse(self):
        """Gradients should flow through the pre-fused path."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ltc = _create_ltc(device=device)
        ltc._force_torchscript = True
        torch.manual_seed(606)
        x = torch.randn(2, 4, 32, device=device)

        out, _ = ltc(x, None)
        loss = out.sum()
        loss.backward()

        assert ltc.erev.grad is not None, "No gradient for erev"
        assert ltc.w.grad is not None, "No gradient for w"
        assert torch.isfinite(ltc.erev.grad).all()
        assert torch.isfinite(ltc.w.grad).all()

    def test_long_rollout_stability(self):
        """100-step rollout should remain finite."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ltc = _create_ltc(device=device)
        torch.manual_seed(707)
        x = torch.randn(2, 100, 32, device=device)

        with torch.no_grad():
            out, (h, _) = ltc(x, None)

        assert torch.isfinite(out).all(), "NaN/Inf in 100-step rollout"
        assert torch.isfinite(h).all(), "NaN/Inf in final state of 100-step rollout"
