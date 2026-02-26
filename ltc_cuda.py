"""Custom CUDA kernels for LTC ODE solver.

Provides fused forward kernels that replace ~2000 individual CUDA kernel launches
with 2 launches per timestep (16 total for T=8). Falls back to TorchScript when
CUDA compilation is not available.

Usage:
    The TrueLiquidTimeConstant module in ltc.py auto-detects and dispatches to
    these kernels when available. No config change needed.
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HAS_CUDA_KERNELS = False
_cuda_module = None
_CUDA_EXT_NAME = "ltc_cuda_kernels_v2"


def _find_cuda_home() -> str | None:
    """Find CUDA toolkit matching PyTorch's CUDA runtime version."""
    pt_cuda = torch.version.cuda  # e.g. "12.6"
    if pt_cuda is None:
        return None
    major_minor = pt_cuda.split(".")[:2]
    version_dir = f"v{major_minor[0]}.{major_minor[1]}"

    base_paths = [
        os.path.join(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", version_dir),
        os.path.join(os.environ.get("CUDA_PATH", ""), ".."),
    ]
    for p in base_paths:
        nvcc = os.path.join(p, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
        if os.path.isfile(nvcc):
            return os.path.abspath(p)
    return None


def _setup_msvc_env() -> str | None:
    """Set up full MSVC environment by running vcvars64.bat and capturing env vars.

    Returns cl.exe directory path, or None if setup failed.
    nvcc internally invokes vcvars64.bat but this often fails in non-interactive
    contexts. By pre-loading the entire VS environment into our process, nvcc
    inherits it and skips the vcvars call.
    """
    import glob
    import subprocess

    # Find vcvars64.bat
    vcvars_patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    vcvars_bat = None
    for pat in vcvars_patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            vcvars_bat = matches[-1]
            break
    if not vcvars_bat:
        return None

    # Run vcvars64.bat and capture resulting environment
    try:
        result = subprocess.run(
            f'cmd /c ""{vcvars_bat}" && set"',
            capture_output=True, text=True, shell=True, timeout=30,
        )
        if result.returncode != 0:
            return None
    except Exception:
        return None

    # Parse the environment and inject into current process
    cl_dir = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key] = value

    # Ensure vswhere.exe is on PATH — nvcc's internal vcvars call needs it
    vswhere_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer"
    if os.path.isdir(vswhere_dir) and vswhere_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = vswhere_dir + os.pathsep + os.environ.get("PATH", "")

    # Find cl.exe from the now-populated PATH
    msvc_patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]
    for pat in msvc_patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            cl_dir = os.path.dirname(matches[-1])
            break

    return cl_dir


def _try_load_cached_pyd(ext_dir: str) -> bool:
    """Try to load a previously compiled .pyd directly, skipping MSVC/nvcc entirely.

    This handles environments where compilation tools aren't available (e.g. under
    ncu profiling, or stripped deployments) but a cached build exists.
    """
    global _HAS_CUDA_KERNELS, _cuda_module
    import importlib.util

    pyd_path = os.path.join(ext_dir, f"{_CUDA_EXT_NAME}.pyd" if sys.platform == "win32"
                            else f"{_CUDA_EXT_NAME}.so")
    if not os.path.isfile(pyd_path):
        return False

    try:
        spec = importlib.util.spec_from_file_location(_CUDA_EXT_NAME, pyd_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cuda_module = mod
        _HAS_CUDA_KERNELS = True
        logger.info("Loaded cached CUDA LTC kernels from %s", pyd_path)
        return True
    except Exception as e:
        logger.debug("Failed to load cached .pyd: %s", e)
        return False


def _try_load_cuda_kernels():
    """Attempt to JIT-compile and load CUDA kernels."""
    global _HAS_CUDA_KERNELS, _cuda_module

    if not torch.cuda.is_available():
        return False

    # Build directory for cached .pyd/.so (inside project directory)
    ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cuda_build")
    os.makedirs(ext_dir, exist_ok=True)

    # Fast path: try loading cached .pyd first, skipping MSVC/nvcc setup entirely.
    # This saves ~6-8 seconds on Windows by avoiding the vcvars64.bat subprocess.
    # Only fall through to compilation if the cached binary doesn't exist or fails.
    kernel_path = os.path.join(os.path.dirname(__file__), "csrc", "ltc_kernels.cu")
    if _try_load_cached_pyd(ext_dir):
        # Verify cached binary is newer than source (skip if source changed)
        if os.path.exists(kernel_path):
            pyd_name = "ltc_cuda_kernels.pyd" if sys.platform == "win32" else "ltc_cuda_kernels.so"
            pyd_path = os.path.join(ext_dir, pyd_name)
            if os.path.getmtime(pyd_path) >= os.path.getmtime(kernel_path):
                logger.info("CUDA LTC kernels loaded from cache (skipped compilation)")
                return True
            else:
                logger.info("CUDA kernel source changed, recompiling...")
                _HAS_CUDA_KERNELS = False
                _cuda_module = None
        else:
            logger.info("CUDA LTC kernels loaded from cache (no source to check)")
            return True

    try:
        from torch.utils.cpp_extension import load

        # Read kernel source
        if not os.path.exists(kernel_path):
            logger.debug("CUDA kernel source not found at %s", kernel_path)
            return False

        extra_cuda_cflags = ["-O3", "--use_fast_math"]
        if sys.platform == "win32":
            # Pre-load full MSVC environment (vcvars64.bat) so nvcc inherits it.
            # --use-local-env tells nvcc to use our pre-loaded env instead of
            # trying to invoke vcvars64.bat internally (which fails from Python).
            # IMPORTANT: must run BEFORE setting CUDA_HOME, because vcvars may
            # overwrite PATH/CUDA_PATH with an older CUDA toolkit version.
            cl_path = _setup_msvc_env()
            if cl_path:
                extra_cuda_cflags.append("--use-local-env")
                logger.info("Using MSVC cl.exe from: %s", cl_path)
            else:
                logger.warning("Could not set up MSVC environment for CUDA compilation")
                return False
        # Point CUDA_HOME to the toolkit matching PyTorch's runtime.
        # Must be set AFTER _setup_msvc_env() which may overwrite CUDA_PATH
        # with an older toolkit version from the system PATH.
        cuda_home = _find_cuda_home()
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
            # Ensure the correct nvcc is found first on PATH
            cuda_bin = os.path.join(cuda_home, "bin")
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
            # PyTorch caches CUDA_HOME at import time — override the module var
            import torch.utils.cpp_extension as _cpp_ext
            _cpp_ext.CUDA_HOME = cuda_home
            logger.info("Using CUDA toolkit: %s", cuda_home)

        # Auto-detect GPU compute capability if not explicitly set
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            cap = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"

        _cuda_module = load(
            name=_CUDA_EXT_NAME,
            sources=[kernel_path],
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=ext_dir,
            verbose=False,
        )
        _HAS_CUDA_KERNELS = True
        logger.info("CUDA LTC kernels compiled and loaded successfully")
        return True

    except Exception as e:
        error_msg = str(e)
        if "Could not set up the environment for Microsoft Visual Studio" in error_msg:
            logger.warning(
                "CUDA kernel compilation failed: nvcc cannot find MSVC. "
                "PyTorch uses CUDA %s runtime. "
                "Fix: install CUDA toolkit %s or run from Developer Command Prompt.",
                torch.version.cuda, torch.version.cuda,
            )
        else:
            logger.warning("Failed to compile CUDA LTC kernels: %s", e)
        # Last resort: try loading cached .pyd even if compilation failed
        return _try_load_cached_pyd(ext_dir)


# Try loading at import time
_try_load_cuda_kernels()


class LTCCudaForward(torch.autograd.Function):
    """Custom autograd function for LTC forward using CUDA kernels.

    Forward uses fused CUDA kernels; backward uses PyTorch autograd
    (via saved tensors and re-computation where needed).
    """

    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,        # (B, T, D)
        state: torch.Tensor,        # (B, H)
        w_pos: torch.Tensor,        # (H, H)
        w_erev: torch.Tensor,       # (H, H)
        mu: torch.Tensor,           # (H, H)
        sigma: torch.Tensor,        # (H, H)
        sensory_mu: torch.Tensor,   # (D, H)
        sensory_sigma: torch.Tensor,# (D, H)
        sensory_erev: torch.Tensor, # (D, H)
        sensory_w_pos: torch.Tensor,# (D, H)
        cm_t: torch.Tensor,         # (H,)
        gleak_pos: torch.Tensor,    # (H,)
        vleak: torch.Tensor,        # (H,)
        ode_unfolds: int,
        epsilon: float,
        needs_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Single C++ call for the entire forward T-loop.
        # Transposes matrices once internally, eliminates Python loop overhead.
        result = _cuda_module.ltc_full_forward(
            x_seq, state,
            w_pos, w_erev, mu, sigma,
            sensory_mu, sensory_sigma, sensory_erev, sensory_w_pos,
            cm_t, gleak_pos, vleak,
            ode_unfolds, epsilon, needs_grad,
        )

        outputs = result[0]       # (B, T, H)
        current_state = result[1]  # (B, H)

        if needs_grad:
            v_buffers_stacked = result[2]     # (T, unfolds, B, H)
            w_num_sens_stacked = result[3]    # (T, B, H)
            w_den_sens_stacked = result[4]    # (T, B, H)
            w_pos_t = result[5]               # (H, H) transposed
            w_erev_t = result[6]              # (H, H) transposed
            mu_t = result[7]                  # (H, H) transposed
            sigma_t = result[8]               # (H, H) transposed
            sensory_mu_t = result[9]          # (H, D) transposed
            sensory_sigma_t = result[10]      # (H, D) transposed
            sensory_erev_t = result[11]       # (H, D) transposed
            sensory_w_pos_t = result[12]      # (H, D) transposed
            ctx.save_for_backward(
                cm_t, gleak_pos, vleak,
                x_seq, state,
                w_pos_t, w_erev_t, mu_t, sigma_t,
                sensory_mu_t, sensory_sigma_t, sensory_erev_t, sensory_w_pos_t,
                v_buffers_stacked, w_num_sens_stacked, w_den_sens_stacked,
            )
        ctx.ode_unfolds = ode_unfolds
        ctx.epsilon = epsilon

        return outputs, current_state

    @staticmethod
    def backward(ctx, grad_outputs, grad_state):
        """Backward pass using C++ T-loop wrapper (Phase 4).

        Single C++ call replaces the entire Python backward loop, eliminating:
        - 11 torch.zeros_like() calls
        - 88 Python-level += operations (11 params × 8 timesteps)
        - Python loop overhead per timestep
        Uses fused backward kernels with per-batch gradient accumulation to
        reduce launch overhead and atomic contention.
        """
        (cm_t, gleak_pos, vleak,
         x_seq, state,
         w_pos_t, w_erev_t, mu_t, sigma_t,
         sensory_mu_t, sensory_sigma_t, sensory_erev_t, sensory_w_pos_t,
         v_buffers_stacked, w_num_sens_stacked, w_den_sens_stacked,
        ) = ctx.saved_tensors

        B, T, D = x_seq.shape
        H = state.shape[1]

        has_grad_state = grad_state is not None
        if not has_grad_state:
            grad_state = torch.zeros(B, H, device=x_seq.device, dtype=x_seq.dtype)

        # Single C++ call for the entire backward pass
        results = _cuda_module.ltc_full_backward(
            x_seq, state,
            w_pos_t, w_erev_t, mu_t, sigma_t,
            sensory_mu_t, sensory_sigma_t, sensory_erev_t, sensory_w_pos_t,
            cm_t, gleak_pos, vleak,
            v_buffers_stacked, w_num_sens_stacked, w_den_sens_stacked,
            grad_outputs, grad_state,
            ctx.ode_unfolds, ctx.epsilon, has_grad_state,
        )

        # Return gradients in the same order as forward inputs
        return (
            results[0],          # grad_x_seq
            results[1],          # grad_state
            results[2],          # grad_w_pos
            results[3],          # grad_w_erev
            results[4],          # grad_mu
            results[5],          # grad_sigma
            results[6],          # grad_sensory_mu
            results[7],          # grad_sensory_sigma
            results[8],          # grad_sensory_erev
            results[9],          # grad_sensory_w_pos
            results[10],         # grad_cm
            results[11],         # grad_gleak
            results[12],         # grad_vleak
            None,                # ode_unfolds (int)
            None,                # epsilon (float)
            None,                # needs_grad (bool)
        )


def ltc_cuda_forward(
    x_seq: torch.Tensor,
    state: torch.Tensor,
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
    """Entry point for CUDA-accelerated LTC forward pass."""
    # Check grad mode HERE (outside Function.forward where it's always disabled)
    needs_grad = torch.is_grad_enabled()
    return LTCCudaForward.apply(
        x_seq, state,
        w_pos, w_erev, mu, sigma,
        sensory_mu, sensory_sigma, sensory_erev, sensory_w_pos,
        cm_t, gleak_pos, vleak,
        ode_unfolds, epsilon, needs_grad,
    )
