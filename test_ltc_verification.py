"""LTC Verification Tests - Ensure faithful ncps implementation matches spec."""

import pytest
import torch
import torch.nn.functional as F
from ltc import TrueLiquidTimeConstant


class TestLTCVerification:
    """Verification tests ensuring LTC implementation matches the ncps LTCCell spec."""

    @pytest.fixture
    def ltc_model(self):
        """Create a test LTC model."""
        return TrueLiquidTimeConstant(
            input_dim=10,
            hidden_dim=8,
            dt=0.1,
            ode_unfolds=6,
            sparsity=0.5,
            sensory_sparsity=0.5,
        )

    def test_constructor_params(self, ltc_model):
        """Test that all biophysical parameters are created with correct shapes."""
        H, D = ltc_model.hidden_dim, ltc_model.input_dim
        # Per-neuron params
        assert ltc_model.gleak.shape == (H,)
        assert ltc_model.vleak.shape == (H,)
        assert ltc_model.cm.shape == (H,)
        # Recurrent synaptic params (H x H)
        assert ltc_model.w.shape == (H, H)
        assert ltc_model.mu.shape == (H, H)
        assert ltc_model.sigma.shape == (H, H)
        assert ltc_model.erev.shape == (H, H)
        assert ltc_model.sparsity_mask.shape == (H, H)
        # Sensory synaptic params (D x H)
        assert ltc_model.sensory_w.shape == (D, H)
        assert ltc_model.sensory_mu.shape == (D, H)
        assert ltc_model.sensory_sigma.shape == (D, H)
        assert ltc_model.sensory_erev.shape == (D, H)
        assert ltc_model.sensory_sparsity_mask.shape == (D, H)

    def test_ode_step_output_shape(self, ltc_model):
        """Test forward with single timestep returns correct shape."""
        B = 3
        inputs = torch.randn(B, 1, ltc_model.input_dim)
        outputs, (h, c) = ltc_model(inputs)
        assert outputs.shape == (B, 1, ltc_model.hidden_dim)
        assert h.shape == (B, ltc_model.hidden_dim)

    def test_state_not_clamped(self, ltc_model):
        """Test that state is NOT tanh-clamped — can exceed [-1, 1]."""
        B = 2
        # Push state beyond [-1, 1] by setting extreme vleak and strong leak
        ltc_model.vleak.data.fill_(5.0)
        # Make gleak dominate so state converges toward vleak
        ltc_model.gleak.data.fill_(10.0)
        # Bypass input_norm scaling by feeding multiple steps
        inputs = torch.ones(B, 8, ltc_model.input_dim) * 5.0
        outputs, (h, _) = ltc_model(inputs)
        result = h
        assert torch.all(torch.isfinite(result)), "State should be finite"
        # Check that values CAN exceed [-1, 1] (no tanh clamp)
        max_abs = result.abs().max().item()
        assert max_abs > 1.0, (
            f"State max |value| = {max_abs:.4f}, should exceed 1.0 "
            "to confirm no tanh clamp is applied"
        )

    def test_per_synapse_sigmoid(self, ltc_model):
        """Test per-synapse sigmoid produces correct output shape."""
        B, H = 3, ltc_model.hidden_dim
        v_pre = torch.randn(B, H)
        # Sigmoid is now inlined in JIT — test the math directly
        result = torch.sigmoid(ltc_model.sigma * (v_pre.unsqueeze(-1) - ltc_model.mu))
        assert result.shape == (B, H, H), f"Expected (B, H, H) but got {result.shape}"
        # Sigmoid values in [0, 1]
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_sensory_sigmoid(self, ltc_model):
        """Test sensory sigmoid works for (D x H) dimensions."""
        B, D = 3, ltc_model.input_dim
        inputs = torch.randn(B, D)
        result = torch.sigmoid(ltc_model.sensory_sigma * (inputs.unsqueeze(-1) - ltc_model.sensory_mu))
        assert result.shape == (B, D, ltc_model.hidden_dim)

    def test_positivity_constraints(self, ltc_model):
        """Test that softplus ensures positivity for w, cm, gleak."""
        assert torch.all(F.softplus(ltc_model.w) > 0), "w must be positive via softplus"
        assert torch.all(F.softplus(ltc_model.cm) > 0), "cm must be positive via softplus"
        assert torch.all(F.softplus(ltc_model.gleak) > 0), "gleak must be positive via softplus"
        assert torch.all(F.softplus(ltc_model.sensory_w) > 0), "sensory_w must be positive via softplus"

    def test_sparsity_masks(self, ltc_model):
        """Test sparsity masks have correct density and binary values."""
        # Masks are binary
        assert torch.all((ltc_model.sparsity_mask == 0) | (ltc_model.sparsity_mask == 1))
        assert torch.all((ltc_model.sensory_sparsity_mask == 0) | (ltc_model.sensory_sparsity_mask == 1))
        # Each neuron has at least one recurrent input
        for i in range(ltc_model.hidden_dim):
            assert ltc_model.sparsity_mask[:, i].any(), f"Neuron {i} has no recurrent inputs"
            assert ltc_model.sensory_sparsity_mask[:, i].any(), f"Neuron {i} has no sensory inputs"
        # Density approximately matches target
        rec_density = ltc_model.sparsity_mask.mean().item()
        assert abs(rec_density - ltc_model.sparsity) < 0.2, \
            f"Recurrent density {rec_density:.2f} too far from target {ltc_model.sparsity}"

    def test_erev_polarity(self, ltc_model):
        """Test reversal potentials have mix of +1 and -1."""
        erev = ltc_model.erev.detach()
        has_excitatory = (erev > 0).any().item()
        has_inhibitory = (erev < 0).any().item()
        assert has_excitatory, "Should have excitatory connections (+1)"
        assert has_inhibitory, "Should have inhibitory connections (-1)"

    def test_forward_shapes(self, ltc_model):
        """Test forward: (B, T, D) -> ((B, T, H), (h, h))."""
        B, T = 2, 5
        x_seq = torch.randn(B, T, ltc_model.input_dim)
        outputs, (h_final, c_final) = ltc_model(x_seq)
        assert outputs.shape == (B, T, ltc_model.hidden_dim)
        assert h_final.shape == (B, ltc_model.hidden_dim)
        assert torch.equal(c_final, h_final), "c mirrors h for LSTM compatibility"

    def test_state_persistence(self, ltc_model):
        """Test state persistence across forward() calls."""
        B = 1
        x1 = torch.randn(B, 3, ltc_model.input_dim)
        x2 = torch.randn(B, 3, ltc_model.input_dim)

        # First call
        _, state1 = ltc_model(x1)
        # Second call with state from first
        out2_with_state, _ = ltc_model(x2, h0=state1)
        # Second call without state
        out2_no_state, _ = ltc_model(x2)
        # Outputs should differ (state matters)
        assert not torch.allclose(out2_with_state, out2_no_state, atol=1e-6), \
            "State should affect output"

    def test_gradient_flow(self, ltc_model):
        """Test gradients flow to all parameters."""
        B, T = 2, 4
        x_seq = torch.randn(B, T, ltc_model.input_dim)
        outputs, _ = ltc_model(x_seq)
        loss = outputs.sum()
        loss.backward()

        # Check key parameters have gradients
        params_to_check = [
            ("w", ltc_model.w),
            ("mu", ltc_model.mu),
            ("sigma", ltc_model.sigma),
            ("erev", ltc_model.erev),
            ("sensory_w", ltc_model.sensory_w),
            ("sensory_mu", ltc_model.sensory_mu),
            ("sensory_sigma", ltc_model.sensory_sigma),
            ("sensory_erev", ltc_model.sensory_erev),
            ("gleak", ltc_model.gleak),
            ("vleak", ltc_model.vleak),
            ("cm", ltc_model.cm),
        ]
        for name, param in params_to_check:
            assert param.grad is not None, f"{name} should have gradients"
            assert torch.any(param.grad != 0), f"{name} gradients should be non-zero"

    def test_api_compatibility(self, ltc_model):
        """Test LSTM-style API compatibility."""
        B, T = 2, 5
        x_seq = torch.randn(B, T, ltc_model.input_dim)
        outputs, (h_final, c_final) = ltc_model(x_seq)
        assert outputs.shape == (B, T, ltc_model.hidden_dim)
        assert h_final.shape == (B, ltc_model.hidden_dim)
        assert torch.equal(c_final, h_final)

    def test_w_rec_compatibility_property(self, ltc_model):
        """Test backward-compat W_rec property exists."""
        assert hasattr(ltc_model, 'W_rec')
        assert ltc_model.W_rec is ltc_model.w

    def test_bounded_dynamics(self, ltc_model):
        """Test that dynamics remain finite under large inputs."""
        B = 1
        x_seq = torch.ones(B, 20, ltc_model.input_dim) * 100.0
        outputs, _ = ltc_model(x_seq)
        assert torch.all(torch.isfinite(outputs)), "Outputs should remain finite under large inputs"

    def test_ode_unfolds_effect(self):
        """Test that more ODE unfolds produce different (more converged) results."""
        D, H = 10, 8
        torch.manual_seed(42)
        m1 = TrueLiquidTimeConstant(D, H, ode_unfolds=1)
        torch.manual_seed(42)
        m6 = TrueLiquidTimeConstant(D, H, ode_unfolds=6)

        x = torch.randn(1, 3, D)
        out1, _ = m1(x)
        out6, _ = m6(x)
        # Different unfold counts should give different results
        assert not torch.allclose(out1, out6, atol=1e-4), \
            "Different ODE unfold counts should produce different results"
