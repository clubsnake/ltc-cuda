from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Auto-detect CUDA kernels at import time
try:
    from ltc_cuda import _HAS_CUDA_KERNELS, ltc_cuda_forward
except ImportError:
    _HAS_CUDA_KERNELS = False
    ltc_cuda_forward = None


@torch.jit.script
def _ltc_ode_step_jit(
    inputs: torch.Tensor,       # (B, D)
    state: torch.Tensor,        # (B, H)
    w_pos: torch.Tensor,        # (H, H) — softplus(w) * sparsity_mask
    w_erev: torch.Tensor,       # (H, H) — w_pos * erev (pre-fused)
    mu: torch.Tensor,           # (H, H)
    sigma: torch.Tensor,        # (H, H)
    sensory_mu: torch.Tensor,   # (D, H)
    sensory_sigma: torch.Tensor,# (D, H)
    sensory_erev: torch.Tensor, # (D, H)
    sensory_w_pos: torch.Tensor,# (D, H) — softplus(sensory_w) * mask
    cm_t: torch.Tensor,         # (H,)
    gleak_pos: torch.Tensor,    # (H,)
    vleak: torch.Tensor,        # (H,)
    ode_unfolds: int,
    epsilon: float,
) -> torch.Tensor:
    """TorchScript-compiled ODE solver for one timestep.

    Fuses the inner unfold loop + sensory computation into a single JIT graph,
    eliminating Python interpreter overhead and enabling op fusion.
    """
    # Sensory contribution (loop-invariant — inputs don't change within unfolds)
    # inputs: (B, D) → (B, D, 1), sensory_mu: (D, H)
    sensory_act = torch.sigmoid(sensory_sigma * (inputs.unsqueeze(-1) - sensory_mu))
    sensory_w_act = sensory_w_pos * sensory_act  # (B, D, H)
    w_num_sensory = (sensory_w_act * sensory_erev).sum(dim=1)  # (B, H)
    w_den_sensory = sensory_w_act.sum(dim=1)  # (B, H)

    v_pre = state
    for _ in range(ode_unfolds):
        # Recurrent conductance: sigmoid gating per synapse
        # v_pre: (B, H) → (B, H, 1), mu: (H, H)
        rec_act = torch.sigmoid(sigma * (v_pre.unsqueeze(-1) - mu))

        # w_erev = w_pos * erev is pre-fused outside the loop,
        # saving one (B, H, H) multiply per unfold iteration
        w_num = (rec_act * w_erev).sum(dim=1) + w_num_sensory  # (B, H)
        w_den = (w_pos * rec_act).sum(dim=1) + w_den_sensory  # (B, H)

        numerator = cm_t * v_pre + gleak_pos * vleak + w_num
        denominator = cm_t + gleak_pos + w_den

        v_pre = numerator / (denominator + epsilon)

    return v_pre


@torch.jit.script
def _ltc_forward_loop_jit(
    x_seq: torch.Tensor,        # (B, T, D)
    state: torch.Tensor,        # (B, H)
    w_pos: torch.Tensor,
    w_erev: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sensory_mu: torch.Tensor,
    sensory_sigma: torch.Tensor,
    sensory_erev: torch.Tensor,
    sensory_w_pos: torch.Tensor,
    cm_t: torch.Tensor,
    gleak_pos: torch.Tensor,
    vleak: torch.Tensor,
    ode_unfolds: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchScript-compiled outer time loop over T timesteps.

    Returns (outputs, final_state).
    """
    B = x_seq.size(0)
    T = x_seq.size(1)
    H = state.size(1)

    outputs = torch.empty(B, T, H, device=x_seq.device, dtype=x_seq.dtype)

    for t in range(T):
        state = _ltc_ode_step_jit(
            x_seq[:, t, :], state,
            w_pos, w_erev, mu, sigma,
            sensory_mu, sensory_sigma, sensory_erev, sensory_w_pos,
            cm_t, gleak_pos, vleak,
            ode_unfolds, epsilon,
        )
        outputs[:, t, :] = state

    return outputs, state


class TrueLiquidTimeConstant(nn.Module):
    """
    Faithful implementation of Liquid Time-Constant Networks matching the
    original ncps LTCCell (Hasani/Lechner).

    Key features matching the original:
    - Per-synapse conductance with learned mu/sigma sigmoid gates
    - Per-synapse reversal potentials (erev matrix)
    - Leak reversal potential (vleak) — neurons have non-zero resting state
    - Membrane capacitance (cm) + leak conductance (gleak) replace static tau
    - Sensory synapses — input pathway uses per-synapse conductance
    - 6 ODE unfolds per timestep for convergence
    - NO tanh on state — dynamics are self-bounding through conductance formulation
    - Positivity constraints via softplus on w, cm, gleak

    Performance: Both the inner ODE unfold loop and outer time loop are
    TorchScript-compiled, eliminating Python overhead and enabling JIT fusion.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dt: float = 1.0 / 30.0,
        ode_unfolds: int = 6,
        sparsity: float = 0.5,
        sensory_sparsity: float = 0.5,
        input_mapping: str = "affine",
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = float(dt)
        self.ode_unfolds = int(max(1, ode_unfolds))
        self.sparsity = sparsity
        self.sensory_sparsity = sensory_sparsity
        self.input_mapping = input_mapping
        self.epsilon = epsilon

        # --- Recurrent synaptic parameters (H x H) ---
        self.w = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(0.001, 1.0))
        self.mu = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(3.0, 8.0))

        # Recurrent sparsity mask and reversal potentials
        sparsity_mask, erev = self._create_recurrent_wiring(hidden_dim, sparsity)
        self.register_buffer("sparsity_mask", sparsity_mask)
        self.erev = nn.Parameter(erev)

        # --- Sensory synaptic parameters (D x H) ---
        self.sensory_w = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(0.001, 1.0))
        self.sensory_mu = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(3.0, 8.0))

        # Sensory sparsity mask and reversal potentials
        sensory_mask, sensory_erev = self._create_sensory_wiring(input_dim, hidden_dim, sensory_sparsity)
        self.register_buffer("sensory_sparsity_mask", sensory_mask)
        self.sensory_erev = nn.Parameter(sensory_erev)

        # --- Per-neuron biophysical parameters ---
        self.gleak = nn.Parameter(torch.empty(hidden_dim).uniform_(0.001, 1.0))
        self.vleak = nn.Parameter(torch.empty(hidden_dim).uniform_(-0.2, 0.2))
        self.cm = nn.Parameter(torch.empty(hidden_dim).uniform_(0.4, 0.6))

        # --- Input mapping (affine preprocessing of sensory input) ---
        if input_mapping == "affine":
            self.input_w = nn.Parameter(torch.ones(input_dim))
            self.input_b = nn.Parameter(torch.zeros(input_dim))
        elif input_mapping == "linear":
            self.input_w = nn.Parameter(torch.ones(input_dim))
            self.register_buffer("input_b", torch.zeros(input_dim))
        else:
            self.register_buffer("input_w", torch.ones(input_dim))
            self.register_buffer("input_b", torch.zeros(input_dim))

        # LayerNorm on input: ensures CNN output is well-scaled before entering LTC
        self.input_norm = nn.LayerNorm(input_dim)

        # Inference cache: softplus transforms are expensive to recompute every call
        # but only change after optimizer.step(). Cache them in eval mode.
        self._cached_transforms: dict | None = None

    @staticmethod
    def _create_recurrent_wiring(hidden_dim: int, sparsity: float):
        """Create sparse recurrent connectivity mask and reversal potentials."""
        if sparsity >= 1.0:
            mask = torch.ones(hidden_dim, hidden_dim)
        else:
            mask = (torch.rand(hidden_dim, hidden_dim) < sparsity).float()
            # Ensure each neuron has at least one input
            for i in range(hidden_dim):
                if not mask[:, i].any():
                    j = torch.randint(0, hidden_dim, (1,))
                    mask[j, i] = 1.0

        # Reversal potentials: ~80% excitatory (+1), ~20% inhibitory (-1)
        erev = torch.where(torch.rand(hidden_dim, hidden_dim) < 0.8,
                           torch.ones(hidden_dim, hidden_dim),
                           -torch.ones(hidden_dim, hidden_dim))
        # Zero out where mask is 0
        erev = erev * mask
        return mask, erev

    @staticmethod
    def _create_sensory_wiring(input_dim: int, hidden_dim: int, sparsity: float):
        """Create sparse sensory connectivity mask and reversal potentials."""
        if sparsity >= 1.0:
            mask = torch.ones(input_dim, hidden_dim)
        else:
            mask = (torch.rand(input_dim, hidden_dim) < sparsity).float()
            # Ensure each neuron gets at least one sensory input
            for i in range(hidden_dim):
                if not mask[:, i].any():
                    j = torch.randint(0, input_dim, (1,))
                    mask[j, i] = 1.0

        # Sensory reversal potentials: ~80% excitatory
        erev = torch.where(torch.rand(input_dim, hidden_dim) < 0.8,
                           torch.ones(input_dim, hidden_dim),
                           -torch.ones(input_dim, hidden_dim))
        erev = erev * mask
        return mask, erev

    def forward(
        self,
        x_seq: torch.Tensor,
        h0: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence with LTC dynamics.

        Args:
            x_seq: (B, T, D) input sequence (raw encoder output)
            h0: initial state (h, c) — uses h as initial LTC state

        Returns:
            outputs: (B, T, H) hidden states
            final_state: (h_final, h_final) for LSTM API compatibility
        """
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        # Initialize state
        state = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        if h0 is not None:
            state = h0[0]

        # Normalize input (LayerNorm forces float32 under AMP)
        x_seq = self.input_norm(x_seq)

        # Sync state dtype — x_seq may have been cast by input_norm
        if state.dtype != x_seq.dtype:
            state = state.to(x_seq.dtype)

        # Apply input mapping (affine/linear/none)
        x_seq = x_seq * self.input_w + self.input_b

        # Pre-compute positivity transforms ONCE for all T timesteps.
        # In eval mode, cache these since params don't change between calls.
        if not self.training and self._cached_transforms is not None:
            w_pos, w_erev, cm_t, gleak_pos, sensory_w_pos = (
                self._cached_transforms['w_pos'],
                self._cached_transforms['w_erev'],
                self._cached_transforms['cm_t'],
                self._cached_transforms['gleak_pos'],
                self._cached_transforms['sensory_w_pos'],
            )
        else:
            w_pos = F.softplus(self.w) * self.sparsity_mask
            w_erev = w_pos * self.erev
            cm_t = F.softplus(self.cm) / (self.dt / self.ode_unfolds)
            gleak_pos = F.softplus(self.gleak)
            sensory_w_pos = F.softplus(self.sensory_w) * self.sensory_sparsity_mask
            if not self.training:
                self._cached_transforms = {
                    'w_pos': w_pos, 'w_erev': w_erev, 'cm_t': cm_t,
                    'gleak_pos': gleak_pos, 'sensory_w_pos': sensory_w_pos,
                }

        # Dispatch to CUDA kernels (fused forward + backward).
        # Forward: 11.7x speedup (16 launches vs ~2000).
        # Backward: custom CUDA kernels eliminate TorchScript JIT overhead.
        use_cuda = (
            _HAS_CUDA_KERNELS
            and x_seq.is_cuda
            and x_seq.dtype == torch.float32
            and not getattr(self, '_force_torchscript', False)
        )

        if use_cuda:
            outputs, state = ltc_cuda_forward(
                x_seq, state,
                w_pos, w_erev, self.mu, self.sigma,
                self.sensory_mu, self.sensory_sigma, self.sensory_erev, sensory_w_pos,
                cm_t, gleak_pos, self.vleak,
                self.ode_unfolds, self.epsilon,
            )
        else:
            if not getattr(self, '_ts_fallback_warned', False):
                logger.warning(
                    "LTC using TorchScript fallback (5-8x slower): "
                    "has_cuda=%s, is_cuda=%s, dtype=%s, force_ts=%s",
                    _HAS_CUDA_KERNELS, x_seq.is_cuda, x_seq.dtype,
                    getattr(self, '_force_torchscript', False),
                )
                self._ts_fallback_warned = True
            outputs, state = _ltc_forward_loop_jit(
                x_seq, state,
                w_pos, w_erev, self.mu, self.sigma,
                self.sensory_mu, self.sensory_sigma, self.sensory_erev, sensory_w_pos,
                cm_t, gleak_pos, self.vleak,
                self.ode_unfolds, self.epsilon,
            )

        # Return LSTM-compatible format
        return outputs, (state, state)

    def train(self, mode: bool = True):
        """Override to invalidate softplus cache when switching to train mode."""
        if mode:
            self._cached_transforms = None
        return super().train(mode)

    # --- Compatibility properties for detection heuristics ---

    @property
    def W_rec(self) -> nn.Parameter:
        """Compatibility: returns recurrent weight parameter for hasattr checks."""
        return self.w

    def fused_euler_step(self, *args, **kwargs):
        """Compatibility: wraps ODE solver for hasattr checks in train_supervised.py."""
        raise NotImplementedError(
            "fused_euler_step is removed. Use _ode_solver(inputs, state) instead."
        )


# Backward compatibility alias
TemporalLTC = TrueLiquidTimeConstant
