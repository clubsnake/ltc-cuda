/*
 * Custom CUDA kernels for LTC (Liquid Time-Constant) ODE solver.
 *
 * Two fused kernels replace ~2000 individual CUDA kernel launches per forward pass:
 *   1. ltc_sensory_fwd — fuses sensory synapse computation (loop-invariant)
 *   2. ltc_ode_step_fwd — fuses one ODE unfold iteration
 *
 * Backward kernels are split into specialized passes to reduce register pressure:
 *   3. ltc_sensory_bwd — sensory backward with struct-packed output pointers
 *   4. ltc_ode_step_bwd_state — state gradient + scalar param grads (saves intermediates)
 *   5. ltc_ode_step_bwd_params — per-synapse param grads from intermediates
 *
 * ltc_full_backward — C++ T-loop wrapper eliminating Python overhead
 *
 * Register pressure results (sm_61):
 *   sensory_fwd:       48 regs, 62.5% occupancy
 *   ode_step_fwd:      40 regs, 100% occupancy
 *   sensory_bwd:       26 regs, 100% occupancy
 *   ode_step_bwd_state: 56 regs, 56% occupancy
 *   ode_step_bwd_params: 31 regs, 100% occupancy
 *
 * Target: GTX 1080 (sm_61, 20 SMs, 48KB shared mem/SM, 2MB L2)
 * Dimensions: B=24, D=224, H=64, unfolds=6
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr unsigned WARP_MASK = 0xffffffffu;

// Helper: transpose a 2D matrix and return contiguous copy.
// Used to convert (H,H) row-major → column-major for coalesced warp access.
// Cost: ~16KB for 64x64, ~56KB for 224x64 — negligible (once per fwd/bwd).
static inline torch::Tensor transpose_2d(const torch::Tensor& m) {
    return m.t().contiguous();
}

// ============================================================================
// Kernel 1: Sensory Forward (stride-aware)
// ============================================================================
// Fuses: sigmoid(sensory_sigma * (inputs - sensory_mu)) * softplus(sensory_w) * mask
// Then reduces over D to produce w_num_sensory and w_den_sensory.
//
// Thread layout: block=(32, BLOCK_H), grid=(ceil(H/BLOCK_H), B)
// Each warp handles one (b, h) pair, reduces over d=D in lane-strides.

template <int BLOCK_H>
__launch_bounds__(32 * BLOCK_H, 2)
__global__ void ltc_sensory_fwd_kernel(
    const float* __restrict__ inputs,          // (B, D) or strided (B, ?, D)
    const float* __restrict__ sensory_sigma,   // (D, H)
    const float* __restrict__ sensory_mu,      // (D, H)
    const float* __restrict__ sensory_w_pos,   // (D, H) — already softplus(w)*mask
    const float* __restrict__ sensory_erev,    // (D, H)
    float* __restrict__ w_num_sensory,         // (B, H) output
    float* __restrict__ w_den_sensory,         // (B, H) output
    int B, int D, int H,
    int input_stride                           // stride between batch rows (D if contiguous)
) {
    const int h = blockIdx.x * BLOCK_H + threadIdx.y;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31

    if (b >= B || h >= H) return;

    float sum_num = 0.0f;
    float sum_den = 0.0f;

    const float* inp_b = inputs + b * input_stride;

    for (int d = lane; d < D; d += 32) {
        float x = __ldg(&inp_b[d]);
        int idx = h * D + d;  // transposed (H, D) layout — coalesced warp access

        float sig = __ldg(&sensory_sigma[idx]);
        float mu = __ldg(&sensory_mu[idx]);
        float w = __ldg(&sensory_w_pos[idx]);
        float erev = __ldg(&sensory_erev[idx]);

        float z = sig * (x - mu);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));
        float w_act = w * act;

        sum_num += w_act * erev;
        sum_den += w_act;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_num += __shfl_down_sync(WARP_MASK, sum_num, offset);
        sum_den += __shfl_down_sync(WARP_MASK, sum_den, offset);
    }

    if (lane == 0) {
        w_num_sensory[b * H + h] = sum_num;
        w_den_sensory[b * H + h] = sum_den;
    }
}


// ============================================================================
// Kernel 2: Single ODE Unfold Step (Forward)
// ============================================================================
// Computes ONE unfold iteration of the recurrent ODE step.
// Called 6 times from C++ wrapper with implicit global sync between launches.
//
// Per step: sigmoid(sigma * (v_pre - mu)) * w_pos → reduce over H → state update
//
// Thread layout: block=(32, BLOCK_J), grid=(ceil(H/BLOCK_J), B)
// Each warp handles one (b, j) output neuron, reduces over i=H in lane-strides.

template <int BLOCK_J>
__launch_bounds__(32 * BLOCK_J, 2)
__global__ void ltc_ode_step_fwd_kernel(
    float* __restrict__ state,                  // (B, H) — in/out
    const float* __restrict__ w_pos,            // (H, H)
    const float* __restrict__ w_erev,           // (H, H)
    const float* __restrict__ mu,               // (H, H)
    const float* __restrict__ sigma,            // (H, H)
    const float* __restrict__ w_num_sensory,    // (B, H)
    const float* __restrict__ w_den_sensory,    // (B, H)
    const float* __restrict__ cm_t,             // (H,)
    const float* __restrict__ gleak_pos,        // (H,)
    const float* __restrict__ vleak,            // (H,)
    float* __restrict__ v_save,                 // (B, H) or nullptr — save pre-step state
    int B, int H, float epsilon
) {
    const int j = blockIdx.x * BLOCK_J + threadIdx.y;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (b >= B) return;

    // ALL threads must participate in shared memory load + sync,
    // even if their j >= H (last block may have partial warps).
    extern __shared__ float v_shared[];
    for (int idx = threadIdx.y * 32 + lane; idx < H; idx += BLOCK_J * 32) {
        v_shared[idx] = state[b * H + idx];
    }
    __syncthreads();

    // Now safe to exit for out-of-range j
    if (j >= H) return;

    float cm_j = __ldg(&cm_t[j]);
    float gleak_j = __ldg(&gleak_pos[j]);
    float vleak_j = __ldg(&vleak[j]);
    float w_num_sens_j = __ldg(&w_num_sensory[b * H + j]);
    float w_den_sens_j = __ldg(&w_den_sensory[b * H + j]);

    // Save pre-step state for backward if requested
    if (v_save != nullptr && lane == 0) {
        v_save[b * H + j] = v_shared[j];
    }

    float v_pre_j = v_shared[j];

    // Reduce over i dimension (recurrent synapses into neuron j)
    float local_num = 0.0f;
    float local_den = 0.0f;

    for (int i = lane; i < H; i += 32) {
        int idx = j * H + i;  // transposed (H, H) — coalesced warp access
        float v_i = v_shared[i];
        float sig = __ldg(&sigma[idx]);
        float m = __ldg(&mu[idx]);

        float z = sig * (v_i - m);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));

        local_num += act * __ldg(&w_erev[idx]);
        local_den += __ldg(&w_pos[idx]) * act;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_num += __shfl_down_sync(WARP_MASK, local_num, offset);
        local_den += __shfl_down_sync(WARP_MASK, local_den, offset);
    }

    if (lane == 0) {
        float w_num = local_num + w_num_sens_j;
        float w_den = local_den + w_den_sens_j;
        float numerator = cm_j * v_pre_j + gleak_j * vleak_j + w_num;
        float denominator = cm_j + gleak_j + w_den;
        state[b * H + j] = numerator / (denominator + epsilon);
    }
}


// ============================================================================
// Kernel 3: Sensory Backward (stride-aware, struct-packed outputs)
// ============================================================================
// Thread layout: block=(32, BLOCK_H), grid=(ceil(H/BLOCK_H), B)
// Parameter grads accumulated across batch with atomicAdd.
// minBlocks=8 forces register count to ~26 (100% occupancy on sm_61).

struct SensoryGradPtrs {
    float* grad_sigma;    // (D, H)
    float* grad_mu;       // (D, H)
    float* grad_w_pos;    // (D, H)
    float* grad_erev;     // (D, H)
};

template <int BLOCK_H>
__launch_bounds__(32 * BLOCK_H, 8)
__global__ void ltc_sensory_bwd_kernel(
    const float* __restrict__ grad_w_num,     // (B, H)
    const float* __restrict__ grad_w_den,     // (B, H)
    const float* __restrict__ inputs,         // (B, D) or strided
    const float* __restrict__ sensory_sigma,  // (D, H)
    const float* __restrict__ sensory_mu,     // (D, H)
    const float* __restrict__ sensory_w_pos,  // (D, H)
    const float* __restrict__ sensory_erev,   // (D, H)
    float* __restrict__ grad_inputs,          // (B, D) — output (contiguous)
    const SensoryGradPtrs grad_ptrs,          // struct with 4 output pointers
    int B, int D, int H,
    int input_stride                          // stride between batch rows for inputs
) {
    const int h = blockIdx.x * BLOCK_H + threadIdx.y;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (b >= B || h >= H) return;

    // Load struct pointers once into registers
    float* const g_sigma = grad_ptrs.grad_sigma;
    float* const g_mu = grad_ptrs.grad_mu;
    float* const g_w_pos = grad_ptrs.grad_w_pos;
    float* const g_erev = grad_ptrs.grad_erev;

    float dw_num = grad_w_num[b * H + h];
    float dw_den = grad_w_den[b * H + h];

    const float* inp_b = inputs + b * input_stride;

    #pragma unroll 1
    for (int d = lane; d < D; d += 32) {
        int idx = h * D + d;  // transposed (H, D) — coalesced warp access

        float x = __ldg(&inp_b[d]);
        float sig = __ldg(&sensory_sigma[idx]);
        float m = __ldg(&sensory_mu[idx]);
        float w = __ldg(&sensory_w_pos[idx]);
        float erev = __ldg(&sensory_erev[idx]);

        // Forward recomputation
        float z = sig * (x - m);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));
        float act_deriv = act * (1.0f - act);
        float w_act = w * act;

        float dw_act = dw_num * erev + dw_den;
        float dact = dw_act * w;
        float dz = dact * act_deriv;

        // Parameter gradients (accumulated across batch)
        atomicAdd(&g_sigma[idx], dz * (x - m));
        atomicAdd(&g_mu[idx], dz * (-sig));
        atomicAdd(&g_w_pos[idx], dw_act * act);
        atomicAdd(&g_erev[idx], dw_num * w_act);

        // Input gradient
        atomicAdd(&grad_inputs[b * D + d], dz * sig);
    }
}


// ============================================================================
// Kernel 4: ODE Step Backward — Pass 1: State gradient + scalar param grads
// ============================================================================
// Computes: grad_v_pre, grad_w_num/den_sensory, grad_cm, grad_gleak, grad_vleak
// Also saves dw_num_per_j and dw_den_per_j to intermediate (B,H) buffers for Pass 2.
//
// Shared memory layout: [0..H-1] = v_pre, [H..4H-1] = cm, gleak, vleak
// Thread layout: block=(32, BLOCK_J), grid=(ceil(H/BLOCK_J), B)

template <int BLOCK_J>
__launch_bounds__(32 * BLOCK_J, 8)
__global__ void ltc_ode_step_bwd_state_kernel(
    const float* __restrict__ v_pre,          // (B, H)
    const float* __restrict__ w_pos,          // (H, H)
    const float* __restrict__ w_erev,         // (H, H)
    const float* __restrict__ mu,             // (H, H)
    const float* __restrict__ sigma,          // (H, H)
    const float* __restrict__ w_num_sensory,  // (B, H)
    const float* __restrict__ w_den_sensory,  // (B, H)
    const float* __restrict__ cm_t,           // (H,)
    const float* __restrict__ gleak_pos,      // (H,)
    const float* __restrict__ vleak,          // (H,)
    const float* __restrict__ grad_v_new,     // (B, H)
    float* __restrict__ grad_v_pre,           // (B, H) — output
    float* __restrict__ grad_w_num_sensory,   // (B, H) — accumulated
    float* __restrict__ grad_w_den_sensory,   // (B, H) — accumulated
    float* __restrict__ grad_cm,              // (H,) — atomicAdd
    float* __restrict__ grad_gleak,           // (H,) — atomicAdd
    float* __restrict__ grad_vleak,           // (H,) — atomicAdd
    float* __restrict__ dw_num_buf,           // (B, H) — intermediate for Pass 2
    float* __restrict__ dw_den_buf,           // (B, H) — intermediate for Pass 2
    int B, int H, float epsilon
) {
    const int j = blockIdx.x * BLOCK_J + threadIdx.y;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (b >= B) return;

    // Shared memory for v_pre + prologue scalars (Phase 3)
    extern __shared__ float smem[];
    float* v_shared = smem;
    float* cm_shared = smem + H;
    float* gleak_shared = smem + 2 * H;
    float* vleak_shared = smem + 3 * H;

    // Cooperative load
    for (int idx = threadIdx.y * 32 + lane; idx < H; idx += BLOCK_J * 32) {
        v_shared[idx] = v_pre[b * H + idx];
        cm_shared[idx] = __ldg(&cm_t[idx]);
        gleak_shared[idx] = __ldg(&gleak_pos[idx]);
        vleak_shared[idx] = __ldg(&vleak[idx]);
    }
    __syncthreads();

    if (j >= H) return;

    float cm_j = cm_shared[j];
    float gleak_j = gleak_shared[j];
    float vleak_j = vleak_shared[j];
    float w_num_sens_j = __ldg(&w_num_sensory[b * H + j]);
    float w_den_sens_j = __ldg(&w_den_sensory[b * H + j]);
    float v_pre_j = v_shared[j];

    // Recompute forward to get w_num, w_den, v_new, denom
    float local_num = 0.0f;
    float local_den = 0.0f;
    for (int i = lane; i < H; i += 32) {
        int idx = j * H + i;  // transposed (H, H) — coalesced
        float v_i = v_shared[i];
        float sig = __ldg(&sigma[idx]);
        float m = __ldg(&mu[idx]);
        float z = sig * (v_i - m);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));
        local_num += act * __ldg(&w_erev[idx]);
        local_den += __ldg(&w_pos[idx]) * act;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_num += __shfl_down_sync(WARP_MASK, local_num, offset);
        local_den += __shfl_down_sync(WARP_MASK, local_den, offset);
    }

    // Broadcast reduced values to all lanes
    float w_num_j = __shfl_sync(WARP_MASK, local_num, 0) + w_num_sens_j;
    float w_den_j = __shfl_sync(WARP_MASK, local_den, 0) + w_den_sens_j;
    float denom = cm_j + gleak_j + w_den_j + epsilon;
    float v_new = (cm_j * v_pre_j + gleak_j * vleak_j + w_num_j) / denom;

    // Backward through the state update
    float dv = grad_v_new[b * H + j];
    float inv_denom = 1.0f / denom;
    float dw_num = dv * inv_denom;
    float dw_den = -dv * v_new * inv_denom;

    // Per-neuron param grads + save intermediates (only lane 0)
    if (lane == 0) {
        grad_w_num_sensory[b * H + j] += dw_num;
        grad_w_den_sensory[b * H + j] += dw_den;

        atomicAdd(&grad_cm[j], dv * (v_pre_j - v_new) * inv_denom);
        atomicAdd(&grad_gleak[j], dv * (vleak_j - v_new) * inv_denom);
        atomicAdd(&grad_vleak[j], dv * gleak_j * inv_denom);

        // Save intermediates for Pass 2 (param grads)
        dw_num_buf[b * H + j] = dw_num;
        dw_den_buf[b * H + j] = dw_den;
    }

    // Gradient through recurrent synapses → scatter to v_pre[i]
    // (param grads deferred to Pass 2)
    for (int i = lane; i < H; i += 32) {
        int idx = j * H + i;  // transposed (H, H) — coalesced
        float v_i = v_shared[i];
        float sig = __ldg(&sigma[idx]);
        float m = __ldg(&mu[idx]);
        float z = sig * (v_i - m);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));
        float act_deriv = act * (1.0f - act);

        float w_e = __ldg(&w_erev[idx]);
        float w_p = __ldg(&w_pos[idx]);
        float dact = dw_num * w_e + dw_den * w_p;
        float dz = dact * act_deriv;
        float dv_i = dz * sig;

        atomicAdd(&grad_v_pre[b * H + i], dv_i);
    }

    // cm path gradient to v_pre_j
    if (lane == 0) {
        atomicAdd(&grad_v_pre[b * H + j], dv * cm_j * inv_denom);
    }
}


// ============================================================================
// Kernel 5: ODE Step Backward — Pass 2: Per-synapse parameter grads
// ============================================================================
// Reads dw_num_per_j and dw_den_per_j from intermediate buffers (no forward
// recompute for dw needed!). Only recomputes activation for the i-loop.
// minBlocks=16 forces register count to ~31 (100% occupancy on sm_61).
//
// Thread layout: block=(32, BLOCK_J), grid=(ceil(H/BLOCK_J), B)

template <int BLOCK_J>
__launch_bounds__(32 * BLOCK_J, 16)
__global__ void ltc_ode_step_bwd_params_kernel(
    const float* __restrict__ v_pre,          // (B, H)
    const float* __restrict__ mu,             // (H, H)
    const float* __restrict__ sigma,          // (H, H)
    const float* __restrict__ w_erev,         // (H, H)
    const float* __restrict__ w_pos,          // (H, H)
    const float* __restrict__ dw_num_buf,     // (B, H) — from Pass 1
    const float* __restrict__ dw_den_buf,     // (B, H) — from Pass 1
    float* __restrict__ grad_w_pos,           // (H, H) — atomicAdd
    float* __restrict__ grad_w_erev,          // (H, H) — atomicAdd
    float* __restrict__ grad_mu,              // (H, H) — atomicAdd
    float* __restrict__ grad_sigma,           // (H, H) — atomicAdd
    int B, int H
) {
    const int j = blockIdx.x * BLOCK_J + threadIdx.y;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (b >= B) return;

    // Load v_pre into shared memory
    extern __shared__ float v_shared[];
    for (int idx = threadIdx.y * 32 + lane; idx < H; idx += BLOCK_J * 32) {
        v_shared[idx] = v_pre[b * H + idx];
    }
    __syncthreads();

    if (j >= H) return;

    float dw_num = __ldg(&dw_num_buf[b * H + j]);
    float dw_den = __ldg(&dw_den_buf[b * H + j]);

    // Scatter param grads over i dimension
    #pragma unroll 1
    for (int i = lane; i < H; i += 32) {
        int idx = j * H + i;  // transposed (H, H) — coalesced reads + writes
        float v_i = v_shared[i];
        float sig = __ldg(&sigma[idx]);
        float m = __ldg(&mu[idx]);
        float z = sig * (v_i - m);
        float act = __fdividef(1.0f, 1.0f + __expf(-z));
        float act_deriv = act * (1.0f - act);

        float w_e = __ldg(&w_erev[idx]);
        float w_p = __ldg(&w_pos[idx]);
        float dact = dw_num * w_e + dw_den * w_p;
        float dz = dact * act_deriv;

        atomicAdd(&grad_w_erev[idx], dw_num * act);
        atomicAdd(&grad_w_pos[idx], dw_den * act);
        atomicAdd(&grad_mu[idx], dz * (-sig));
        atomicAdd(&grad_sigma[idx], dz * (v_i - m));
    }
}


// ============================================================================
// C++ wrapper functions
// ============================================================================

std::vector<torch::Tensor> ltc_sensory_fwd(
    torch::Tensor inputs,          // (B, D) or strided
    torch::Tensor sensory_sigma,   // (D, H) — will be transposed for coalescing
    torch::Tensor sensory_mu,      // (D, H)
    torch::Tensor sensory_w_pos,   // (D, H)
    torch::Tensor sensory_erev     // (D, H)
) {
    const int B = inputs.size(0);
    const int D = inputs.size(1);
    const int H = sensory_sigma.size(1);
    const int input_stride = inputs.stride(0);

    // Transpose (D,H) → (H,D) for coalesced lane access
    auto sig_t = transpose_2d(sensory_sigma);
    auto mu_t = transpose_2d(sensory_mu);
    auto w_t = transpose_2d(sensory_w_pos);
    auto erev_t = transpose_2d(sensory_erev);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(inputs.device());
    auto w_num_sensory = torch::empty({B, H}, opts);
    auto w_den_sensory = torch::empty({B, H}, opts);

    constexpr int BLOCK_H = 4;
    dim3 block(32, BLOCK_H);
    dim3 grid((H + BLOCK_H - 1) / BLOCK_H, B);

    ltc_sensory_fwd_kernel<BLOCK_H><<<grid, block>>>(
        inputs.data_ptr<float>(),
        sig_t.data_ptr<float>(),
        mu_t.data_ptr<float>(),
        w_t.data_ptr<float>(),
        erev_t.data_ptr<float>(),
        w_num_sensory.data_ptr<float>(),
        w_den_sensory.data_ptr<float>(),
        B, D, H, input_stride
    );

    return {w_num_sensory, w_den_sensory};
}


std::vector<torch::Tensor> ltc_ode_unfold_fwd(
    torch::Tensor state,           // (B, H) — will be modified in-place, return copy
    torch::Tensor w_pos,           // (H, H) — will be transposed for coalescing
    torch::Tensor w_erev,          // (H, H)
    torch::Tensor mu,              // (H, H)
    torch::Tensor sigma,           // (H, H)
    torch::Tensor w_num_sensory,   // (B, H)
    torch::Tensor w_den_sensory,   // (B, H)
    torch::Tensor cm_t,            // (H,)
    torch::Tensor gleak_pos,       // (H,)
    torch::Tensor vleak,           // (H,)
    int ode_unfolds,
    float epsilon,
    bool save_intermediates
) {
    const int B = state.size(0);
    const int H = state.size(1);

    // Transpose (H,H) → transposed layout for coalesced lane access
    auto wp_t = transpose_2d(w_pos);
    auto we_t = transpose_2d(w_erev);
    auto mu_t = transpose_2d(mu);
    auto sig_t = transpose_2d(sigma);

    auto state_work = state.clone();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(state.device());

    // v_buffer: (unfolds, B, H) — contiguous (B,H) slices
    torch::Tensor v_buffer;
    if (save_intermediates) {
        v_buffer = torch::empty({ode_unfolds, B, H}, opts);
    }

    constexpr int BLOCK_J = 2;
    dim3 block(32, BLOCK_J);
    dim3 grid((H + BLOCK_J - 1) / BLOCK_J, B);
    int smem_bytes = H * sizeof(float);

    for (int k = 0; k < ode_unfolds; k++) {
        float* v_save_ptr = nullptr;
        if (save_intermediates) {
            v_save_ptr = v_buffer.data_ptr<float>() + k * B * H;
        }

        ltc_ode_step_fwd_kernel<BLOCK_J><<<grid, block, smem_bytes>>>(
            state_work.data_ptr<float>(),
            wp_t.data_ptr<float>(),
            we_t.data_ptr<float>(),
            mu_t.data_ptr<float>(),
            sig_t.data_ptr<float>(),
            w_num_sensory.data_ptr<float>(),
            w_den_sensory.data_ptr<float>(),
            cm_t.data_ptr<float>(),
            gleak_pos.data_ptr<float>(),
            vleak.data_ptr<float>(),
            v_save_ptr,
            B, H, epsilon
        );
    }

    return {state_work, v_buffer};
}


std::vector<torch::Tensor> ltc_sensory_bwd(
    torch::Tensor grad_w_num,     // (B, H)
    torch::Tensor grad_w_den,     // (B, H)
    torch::Tensor inputs,         // (B, D)
    torch::Tensor sensory_sigma,  // (D, H)
    torch::Tensor sensory_mu,     // (D, H)
    torch::Tensor sensory_w_pos,  // (D, H)
    torch::Tensor sensory_erev    // (D, H)
) {
    const int B = inputs.size(0);
    const int D = inputs.size(1);
    const int H = sensory_sigma.size(1);
    const int input_stride = inputs.stride(0);

    // Transpose (D,H) → (H,D) for coalesced kernel access (idx = h*D+d)
    auto sig_t = transpose_2d(sensory_sigma);
    auto mu_t = transpose_2d(sensory_mu);
    auto w_t = transpose_2d(sensory_w_pos);
    auto erev_t = transpose_2d(sensory_erev);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(inputs.device());
    auto grad_inputs = torch::zeros({B, D}, opts);
    // Grad accumulators in transposed (H,D) layout — kernel atomicAdds to h*D+d
    auto grad_sensory_sigma_t = torch::zeros({H, D}, opts);
    auto grad_sensory_mu_t = torch::zeros({H, D}, opts);
    auto grad_sensory_w_pos_t = torch::zeros({H, D}, opts);
    auto grad_sensory_erev_t = torch::zeros({H, D}, opts);

    SensoryGradPtrs grad_ptrs;
    grad_ptrs.grad_sigma = grad_sensory_sigma_t.data_ptr<float>();
    grad_ptrs.grad_mu = grad_sensory_mu_t.data_ptr<float>();
    grad_ptrs.grad_w_pos = grad_sensory_w_pos_t.data_ptr<float>();
    grad_ptrs.grad_erev = grad_sensory_erev_t.data_ptr<float>();

    constexpr int BLOCK_H = 4;
    dim3 block(32, BLOCK_H);
    dim3 grid((H + BLOCK_H - 1) / BLOCK_H, B);

    ltc_sensory_bwd_kernel<BLOCK_H><<<grid, block>>>(
        grad_w_num.data_ptr<float>(),
        grad_w_den.data_ptr<float>(),
        inputs.data_ptr<float>(),
        sig_t.data_ptr<float>(),
        mu_t.data_ptr<float>(),
        w_t.data_ptr<float>(),
        erev_t.data_ptr<float>(),
        grad_inputs.data_ptr<float>(),
        grad_ptrs,
        B, D, H, input_stride
    );

    // Transpose grads back to original (D,H) layout
    return {grad_inputs,
            grad_sensory_sigma_t.t().contiguous(),
            grad_sensory_mu_t.t().contiguous(),
            grad_sensory_w_pos_t.t().contiguous(),
            grad_sensory_erev_t.t().contiguous()};
}


// ============================================================================
// ltc_ode_unfold_bwd — Split two-pass backward
// ============================================================================
// Kept for backward compatibility with Python-level T-loop in ltc_cuda.py.

std::vector<torch::Tensor> ltc_ode_unfold_bwd(
    torch::Tensor v_buffer,       // (unfolds, B, H)
    torch::Tensor w_pos,          // (H, H)
    torch::Tensor w_erev,         // (H, H)
    torch::Tensor mu,             // (H, H)
    torch::Tensor sigma,          // (H, H)
    torch::Tensor w_num_sensory,  // (B, H)
    torch::Tensor w_den_sensory,  // (B, H)
    torch::Tensor cm_t,           // (H,)
    torch::Tensor gleak_pos,      // (H,)
    torch::Tensor vleak,          // (H,)
    torch::Tensor grad_output,    // (B, H) — dL/d(v_final)
    int ode_unfolds,
    float epsilon
) {
    const int B = v_buffer.size(1);
    const int H = v_buffer.size(2);

    // Transpose (H,H) → coalesced layout for kernels (idx = j*H+i)
    auto wp_t = transpose_2d(w_pos);
    auto we_t = transpose_2d(w_erev);
    auto mu_t = transpose_2d(mu);
    auto sig_t = transpose_2d(sigma);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(v_buffer.device());
    auto grad_w_num_sensory = torch::zeros({B, H}, opts);
    auto grad_w_den_sensory = torch::zeros({B, H}, opts);
    // Grad accumulators in transposed layout — kernel atomicAdds to j*H+i
    auto grad_w_pos_t = torch::zeros({H, H}, opts);
    auto grad_w_erev_t = torch::zeros({H, H}, opts);
    auto grad_mu_t = torch::zeros({H, H}, opts);
    auto grad_sigma_t = torch::zeros({H, H}, opts);
    auto grad_cm = torch::zeros({H}, opts);
    auto grad_gleak = torch::zeros({H}, opts);
    auto grad_vleak = torch::zeros({H}, opts);

    auto dw_num_buf = torch::empty({B, H}, opts);
    auto dw_den_buf = torch::empty({B, H}, opts);

    constexpr int BLOCK_J = 2;
    dim3 block(32, BLOCK_J);
    dim3 grid((H + BLOCK_J - 1) / BLOCK_J, B);
    int smem_state = 4 * H * sizeof(float);
    int smem_params = H * sizeof(float);

    auto grad_v_current = grad_output.clone();
    auto grad_v_buf0 = torch::zeros({B, H}, opts);
    auto grad_v_buf1 = torch::zeros({B, H}, opts);
    float* grad_v_ptrs[2] = {grad_v_buf0.data_ptr<float>(), grad_v_buf1.data_ptr<float>()};
    int buf_idx = 0;

    for (int k = ode_unfolds - 1; k >= 0; k--) {
        float* v_pre_ptr = v_buffer.data_ptr<float>() + k * B * H;
        float* grad_v_pre_ptr = grad_v_ptrs[buf_idx];
        cudaMemsetAsync(grad_v_pre_ptr, 0, B * H * sizeof(float));

        ltc_ode_step_bwd_state_kernel<BLOCK_J><<<grid, block, smem_state>>>(
            v_pre_ptr,
            wp_t.data_ptr<float>(),
            we_t.data_ptr<float>(),
            mu_t.data_ptr<float>(),
            sig_t.data_ptr<float>(),
            w_num_sensory.data_ptr<float>(),
            w_den_sensory.data_ptr<float>(),
            cm_t.data_ptr<float>(),
            gleak_pos.data_ptr<float>(),
            vleak.data_ptr<float>(),
            grad_v_current.data_ptr<float>(),
            grad_v_pre_ptr,
            grad_w_num_sensory.data_ptr<float>(),
            grad_w_den_sensory.data_ptr<float>(),
            grad_cm.data_ptr<float>(),
            grad_gleak.data_ptr<float>(),
            grad_vleak.data_ptr<float>(),
            dw_num_buf.data_ptr<float>(),
            dw_den_buf.data_ptr<float>(),
            B, H, epsilon
        );

        ltc_ode_step_bwd_params_kernel<BLOCK_J><<<grid, block, smem_params>>>(
            v_pre_ptr,
            mu_t.data_ptr<float>(),
            sig_t.data_ptr<float>(),
            we_t.data_ptr<float>(),
            wp_t.data_ptr<float>(),
            dw_num_buf.data_ptr<float>(),
            dw_den_buf.data_ptr<float>(),
            grad_w_pos_t.data_ptr<float>(),
            grad_w_erev_t.data_ptr<float>(),
            grad_mu_t.data_ptr<float>(),
            grad_sigma_t.data_ptr<float>(),
            B, H
        );

        grad_v_current = (buf_idx == 0) ? grad_v_buf0 : grad_v_buf1;
        buf_idx = 1 - buf_idx;
    }

    // Transpose grads back to original (H,H) layout
    return {grad_v_current, grad_w_num_sensory, grad_w_den_sensory,
            grad_w_pos_t.t().contiguous(),
            grad_w_erev_t.t().contiguous(),
            grad_mu_t.t().contiguous(),
            grad_sigma_t.t().contiguous(),
            grad_cm, grad_gleak, grad_vleak};
}


// ============================================================================
// ltc_full_backward — C++ T-loop wrapper
// ============================================================================
// Single C++ call replaces the entire backward pass, eliminating:
// - 11 torch.zeros_like() calls in Python
// - 88 Python-level += operations (11 params × 8 timesteps)
// - Python loop overhead per timestep
// - contiguous() calls via stride-aware sensory kernel

std::vector<torch::Tensor> ltc_full_backward(
    torch::Tensor x_seq,                // (B, T, D)
    torch::Tensor state,                 // (B, H) — initial state
    torch::Tensor w_pos,                 // (H, H)
    torch::Tensor w_erev,               // (H, H)
    torch::Tensor mu,                    // (H, H)
    torch::Tensor sigma,                 // (H, H)
    torch::Tensor sensory_mu,            // (D, H)
    torch::Tensor sensory_sigma,         // (D, H)
    torch::Tensor sensory_erev,          // (D, H)
    torch::Tensor sensory_w_pos,         // (D, H)
    torch::Tensor cm_t,                  // (H,)
    torch::Tensor gleak_pos,             // (H,)
    torch::Tensor vleak,                 // (H,)
    torch::Tensor v_buffers_stacked,     // (T, unfolds, B, H)
    torch::Tensor w_num_sens_stacked,    // (T, B, H)
    torch::Tensor w_den_sens_stacked,    // (T, B, H)
    torch::Tensor grad_outputs,          // (B, T, H)
    torch::Tensor grad_final_state,      // (B, H) or empty
    int ode_unfolds,
    float epsilon,
    bool has_grad_state
) {
    const int B = x_seq.size(0);
    const int T = x_seq.size(1);
    const int D = x_seq.size(2);
    const int H = state.size(1);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x_seq.device());

    // Transpose all matrices for coalesced kernel access
    auto wp_t = transpose_2d(w_pos);        // (H,H) → transposed
    auto we_t = transpose_2d(w_erev);
    auto mu_t = transpose_2d(mu);
    auto sig_t = transpose_2d(sigma);
    auto s_sig_t = transpose_2d(sensory_sigma);  // (D,H) → (H,D)
    auto s_mu_t = transpose_2d(sensory_mu);
    auto s_erev_t = transpose_2d(sensory_erev);
    auto s_w_t = transpose_2d(sensory_w_pos);

    // Allocate all grad accumulators once — kernels atomicAdd directly into these
    // Recurrent grads in transposed layout (kernel writes to j*H+i)
    auto grad_w_pos_acc_t = torch::zeros({H, H}, opts);
    auto grad_w_erev_acc_t = torch::zeros({H, H}, opts);
    auto grad_mu_acc_t = torch::zeros({H, H}, opts);
    auto grad_sigma_acc_t = torch::zeros({H, H}, opts);
    auto grad_cm_acc = torch::zeros({H}, opts);
    auto grad_gleak_acc = torch::zeros({H}, opts);
    auto grad_vleak_acc = torch::zeros({H}, opts);
    // Sensory grads in transposed (H,D) layout (kernel writes to h*D+d)
    auto grad_sensory_mu_acc_t = torch::zeros({H, D}, opts);
    auto grad_sensory_sigma_acc_t = torch::zeros({H, D}, opts);
    auto grad_sensory_erev_acc_t = torch::zeros({H, D}, opts);
    auto grad_sensory_w_pos_acc_t = torch::zeros({H, D}, opts);

    auto grad_x_seq = torch::empty({B, T, D}, opts);
    auto grad_state_acc = has_grad_state
        ? grad_final_state.clone()
        : torch::zeros({B, H}, opts);

    // Intermediate buffers reused across timesteps
    auto dw_num_buf = torch::empty({B, H}, opts);
    auto dw_den_buf = torch::empty({B, H}, opts);
    auto grad_v_buf0 = torch::zeros({B, H}, opts);
    auto grad_v_buf1 = torch::zeros({B, H}, opts);
    auto ode_grad_w_num_sens = torch::empty({B, H}, opts);
    auto ode_grad_w_den_sens = torch::empty({B, H}, opts);
    auto dv_scratch = torch::empty({B, H}, opts);
    auto grad_inputs_scratch = torch::empty({B, D}, opts);

    constexpr int BLOCK_J = 2;
    constexpr int BLOCK_H = 4;
    dim3 ode_block(32, BLOCK_J);
    dim3 ode_grid((H + BLOCK_J - 1) / BLOCK_J, B);
    dim3 sens_block(32, BLOCK_H);
    dim3 sens_grid((H + BLOCK_H - 1) / BLOCK_H, B);
    int smem_state = 4 * H * sizeof(float);
    int smem_params = H * sizeof(float);

    // Sensory backward struct — points at transposed accumulators (set once)
    SensoryGradPtrs sens_grad_ptrs;
    sens_grad_ptrs.grad_sigma = grad_sensory_sigma_acc_t.data_ptr<float>();
    sens_grad_ptrs.grad_mu = grad_sensory_mu_acc_t.data_ptr<float>();
    sens_grad_ptrs.grad_w_pos = grad_sensory_w_pos_acc_t.data_ptr<float>();
    sens_grad_ptrs.grad_erev = grad_sensory_erev_acc_t.data_ptr<float>();

    // Cache transposed pointers for inner loop
    float* w_pos_ptr = wp_t.data_ptr<float>();
    float* w_erev_ptr = we_t.data_ptr<float>();
    float* mu_ptr = mu_t.data_ptr<float>();
    float* sigma_ptr = sig_t.data_ptr<float>();
    float* cm_t_ptr = cm_t.data_ptr<float>();
    float* gleak_pos_ptr = gleak_pos.data_ptr<float>();
    float* vleak_ptr = vleak.data_ptr<float>();
    float* sensory_sigma_ptr = s_sig_t.data_ptr<float>();
    float* sensory_mu_ptr = s_mu_t.data_ptr<float>();
    float* sensory_w_pos_ptr = s_w_t.data_ptr<float>();
    float* sensory_erev_ptr = s_erev_t.data_ptr<float>();
    float* dw_num_buf_ptr = dw_num_buf.data_ptr<float>();
    float* dw_den_buf_ptr = dw_den_buf.data_ptr<float>();
    float* grad_w_pos_ptr = grad_w_pos_acc_t.data_ptr<float>();
    float* grad_w_erev_ptr = grad_w_erev_acc_t.data_ptr<float>();
    float* grad_mu_ptr = grad_mu_acc_t.data_ptr<float>();
    float* grad_sigma_ptr = grad_sigma_acc_t.data_ptr<float>();
    float* grad_cm_ptr = grad_cm_acc.data_ptr<float>();
    float* grad_gleak_ptr = grad_gleak_acc.data_ptr<float>();
    float* grad_vleak_ptr = grad_vleak_acc.data_ptr<float>();
    float* ode_w_num_sens_ptr = ode_grad_w_num_sens.data_ptr<float>();
    float* ode_w_den_sens_ptr = ode_grad_w_den_sens.data_ptr<float>();
    float* grad_v_ptrs[2] = {grad_v_buf0.data_ptr<float>(), grad_v_buf1.data_ptr<float>()};
    float* v_buffers_ptr = v_buffers_stacked.data_ptr<float>();
    float* w_num_sens_ptr = w_num_sens_stacked.data_ptr<float>();
    float* w_den_sens_ptr = w_den_sens_stacked.data_ptr<float>();
    float* x_seq_ptr = x_seq.data_ptr<float>();
    float* dv_scratch_ptr = dv_scratch.data_ptr<float>();
    float* grad_inputs_ptr = grad_inputs_scratch.data_ptr<float>();
    int x_seq_stride = T * D;  // stride between batch rows in x_seq (B, T, D)

    for (int t = T - 1; t >= 0; t--) {
        // dv_t = grad_outputs[:, t, :] + grad_state_acc — into scratch buffer (no alloc)
        auto grad_out_slice = grad_outputs.select(1, t);  // strided view, no alloc
        torch::add_out(dv_scratch, grad_out_slice, grad_state_acc);

        // Direct pointer arithmetic into pre-allocated stacked tensors
        float* v_buf_t_ptr = v_buffers_ptr + t * ode_unfolds * B * H;
        float* w_num_t_ptr = w_num_sens_ptr + t * B * H;
        float* w_den_t_ptr = w_den_sens_ptr + t * B * H;

        // Zero per-timestep sensory accumulators only
        // Params accumulate directly into global accumulators via atomicAdd
        cudaMemsetAsync(ode_w_num_sens_ptr, 0, B * H * sizeof(float));
        cudaMemsetAsync(ode_w_den_sens_ptr, 0, B * H * sizeof(float));

        float* grad_v_current_ptr = dv_scratch_ptr;
        int buf_idx = 0;

        for (int k = ode_unfolds - 1; k >= 0; k--) {
            float* v_pre_ptr = v_buf_t_ptr + k * B * H;
            float* grad_v_pre_ptr = grad_v_ptrs[buf_idx];
            cudaMemsetAsync(grad_v_pre_ptr, 0, B * H * sizeof(float));

            ltc_ode_step_bwd_state_kernel<BLOCK_J><<<ode_grid, ode_block, smem_state>>>(
                v_pre_ptr,
                w_pos_ptr, w_erev_ptr, mu_ptr, sigma_ptr,
                w_num_t_ptr, w_den_t_ptr,
                cm_t_ptr, gleak_pos_ptr, vleak_ptr,
                grad_v_current_ptr,
                grad_v_pre_ptr,
                ode_w_num_sens_ptr, ode_w_den_sens_ptr,
                grad_cm_ptr, grad_gleak_ptr, grad_vleak_ptr,
                dw_num_buf_ptr, dw_den_buf_ptr,
                B, H, epsilon
            );

            ltc_ode_step_bwd_params_kernel<BLOCK_J><<<ode_grid, ode_block, smem_params>>>(
                v_pre_ptr,
                mu_ptr, sigma_ptr, w_erev_ptr, w_pos_ptr,
                dw_num_buf_ptr, dw_den_buf_ptr,
                grad_w_pos_ptr, grad_w_erev_ptr,
                grad_mu_ptr, grad_sigma_ptr,
                B, H
            );

            grad_v_current_ptr = grad_v_ptrs[buf_idx];
            buf_idx = 1 - buf_idx;
        }

        // grad_v_current_ptr = dL/d(state before this timestep)
        grad_state_acc = (grad_v_current_ptr == grad_v_ptrs[0]) ? grad_v_buf0 : grad_v_buf1;

        // Sensory backward — stride-aware, no contiguous() needed
        float* inputs_t_ptr = x_seq_ptr + t * D;
        cudaMemsetAsync(grad_inputs_ptr, 0, B * D * sizeof(float));

        ltc_sensory_bwd_kernel<BLOCK_H><<<sens_grid, sens_block>>>(
            ode_w_num_sens_ptr, ode_w_den_sens_ptr,
            inputs_t_ptr,
            sensory_sigma_ptr, sensory_mu_ptr,
            sensory_w_pos_ptr, sensory_erev_ptr,
            grad_inputs_ptr,
            sens_grad_ptrs,
            B, D, H, x_seq_stride
        );

        // Copy grad_inputs into grad_x_seq[:, t, :] — strided copy
        grad_x_seq.select(1, t).copy_(grad_inputs_scratch);
    }

    // Transpose grads back to original layout before returning
    return {
        grad_x_seq,                                  // x_seq (B, T, D)
        grad_state_acc,                              // state (B, H)
        grad_w_pos_acc_t.t().contiguous(),           // w_pos (H, H)
        grad_w_erev_acc_t.t().contiguous(),          // w_erev (H, H)
        grad_mu_acc_t.t().contiguous(),              // mu (H, H)
        grad_sigma_acc_t.t().contiguous(),           // sigma (H, H)
        grad_sensory_mu_acc_t.t().contiguous(),      // sensory_mu (D, H)
        grad_sensory_sigma_acc_t.t().contiguous(),   // sensory_sigma (D, H)
        grad_sensory_erev_acc_t.t().contiguous(),    // sensory_erev (D, H)
        grad_sensory_w_pos_acc_t.t().contiguous(),   // sensory_w_pos (D, H)
        grad_cm_acc,                                 // cm_t (H,)
        grad_gleak_acc,                              // gleak_pos (H,)
        grad_vleak_acc,                              // vleak (H,)
    };
}


// ============================================================================
// ltc_full_forward — C++ T-loop wrapper for forward pass
// ============================================================================
// Single C++ call replaces the Python T-loop in forward, eliminating:
// - T Python-level function calls per forward pass
// - T contiguous() calls on x_seq slices
// - T torch::Tensor allocations for sensory outputs

std::vector<torch::Tensor> ltc_full_forward(
    torch::Tensor x_seq,                // (B, T, D)
    torch::Tensor state,                 // (B, H)
    torch::Tensor w_pos,                 // (H, H)
    torch::Tensor w_erev,               // (H, H)
    torch::Tensor mu,                    // (H, H)
    torch::Tensor sigma,                 // (H, H)
    torch::Tensor sensory_mu,            // (D, H)
    torch::Tensor sensory_sigma,         // (D, H)
    torch::Tensor sensory_erev,          // (D, H)
    torch::Tensor sensory_w_pos,         // (D, H)
    torch::Tensor cm_t,                  // (H,)
    torch::Tensor gleak_pos,             // (H,)
    torch::Tensor vleak,                 // (H,)
    int ode_unfolds,
    float epsilon,
    bool save_intermediates
) {
    const int B = x_seq.size(0);
    const int T = x_seq.size(1);
    const int D = x_seq.size(2);
    const int H = state.size(1);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x_seq.device());

    // Transpose all matrices for coalesced kernel access
    auto wp_t = transpose_2d(w_pos);        // (H,H) → transposed
    auto we_t = transpose_2d(w_erev);
    auto mu_t = transpose_2d(mu);
    auto sig_t = transpose_2d(sigma);
    auto s_sig_t = transpose_2d(sensory_sigma);  // (D,H) → (H,D)
    auto s_mu_t = transpose_2d(sensory_mu);
    auto s_erev_t = transpose_2d(sensory_erev);
    auto s_w_t = transpose_2d(sensory_w_pos);

    // Output: (B, T, H)
    auto outputs = torch::empty({B, T, H}, opts);
    auto state_work = state.clone();

    // Intermediate buffers for backward (saved if needs_grad)
    torch::Tensor v_buffers_stacked;     // (T, unfolds, B, H)
    torch::Tensor w_num_sens_stacked;    // (T, B, H)
    torch::Tensor w_den_sens_stacked;    // (T, B, H)
    if (save_intermediates) {
        v_buffers_stacked = torch::empty({T, ode_unfolds, B, H}, opts);
        w_num_sens_stacked = torch::empty({T, B, H}, opts);
        w_den_sens_stacked = torch::empty({T, B, H}, opts);
    }

    // Reusable per-timestep sensory outputs
    auto w_num_sensory = torch::empty({B, H}, opts);
    auto w_den_sensory = torch::empty({B, H}, opts);

    // Kernel launch configs
    constexpr int BLOCK_H = 4;
    constexpr int BLOCK_J = 2;
    dim3 sens_block(32, BLOCK_H);
    dim3 sens_grid((H + BLOCK_H - 1) / BLOCK_H, B);
    dim3 ode_block(32, BLOCK_J);
    dim3 ode_grid((H + BLOCK_J - 1) / BLOCK_J, B);
    int ode_smem = H * sizeof(float);

    // Cache transposed pointers
    float* x_seq_ptr = x_seq.data_ptr<float>();
    float* sensory_sigma_ptr = s_sig_t.data_ptr<float>();
    float* sensory_mu_ptr = s_mu_t.data_ptr<float>();
    float* sensory_w_pos_ptr = s_w_t.data_ptr<float>();
    float* sensory_erev_ptr = s_erev_t.data_ptr<float>();
    float* w_num_ptr = w_num_sensory.data_ptr<float>();
    float* w_den_ptr = w_den_sensory.data_ptr<float>();
    float* state_ptr = state_work.data_ptr<float>();
    float* w_pos_ptr = wp_t.data_ptr<float>();
    float* w_erev_ptr = we_t.data_ptr<float>();
    float* mu_ptr = mu_t.data_ptr<float>();
    float* sigma_ptr = sig_t.data_ptr<float>();
    float* cm_t_ptr = cm_t.data_ptr<float>();
    float* gleak_ptr = gleak_pos.data_ptr<float>();
    float* vleak_ptr = vleak.data_ptr<float>();
    int x_seq_stride = T * D;  // stride between batch rows in (B, T, D)

    float* v_buf_ptr = save_intermediates ? v_buffers_stacked.data_ptr<float>() : nullptr;
    float* w_num_stacked_ptr = save_intermediates ? w_num_sens_stacked.data_ptr<float>() : nullptr;
    float* w_den_stacked_ptr = save_intermediates ? w_den_sens_stacked.data_ptr<float>() : nullptr;

    for (int t = 0; t < T; t++) {
        // Sensory forward — stride-aware, pointer arithmetic directly into x_seq
        float* inputs_t_ptr = x_seq_ptr + t * D;

        ltc_sensory_fwd_kernel<BLOCK_H><<<sens_grid, sens_block>>>(
            inputs_t_ptr,
            sensory_sigma_ptr, sensory_mu_ptr,
            sensory_w_pos_ptr, sensory_erev_ptr,
            w_num_ptr, w_den_ptr,
            B, D, H, x_seq_stride
        );

        // Save sensory outputs for backward
        if (save_intermediates) {
            float* w_num_dst = w_num_stacked_ptr + t * B * H;
            float* w_den_dst = w_den_stacked_ptr + t * B * H;
            cudaMemcpyAsync(w_num_dst, w_num_ptr, B * H * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(w_den_dst, w_den_ptr, B * H * sizeof(float),
                            cudaMemcpyDeviceToDevice);
        }

        // ODE unfold loop
        for (int k = 0; k < ode_unfolds; k++) {
            float* v_save_ptr = nullptr;
            if (save_intermediates) {
                v_save_ptr = v_buf_ptr + (t * ode_unfolds + k) * B * H;
            }

            ltc_ode_step_fwd_kernel<BLOCK_J><<<ode_grid, ode_block, ode_smem>>>(
                state_ptr,
                w_pos_ptr, w_erev_ptr, mu_ptr, sigma_ptr,
                w_num_ptr, w_den_ptr,
                cm_t_ptr, gleak_ptr, vleak_ptr,
                v_save_ptr,
                B, H, epsilon
            );
        }

        // Copy state to output[:, t, :] — outputs is (B, T, H) contiguous
        // outputs[b, t, h] = state_work[b, h], strided copy
        outputs.select(1, t).copy_(state_work);
    }

    if (save_intermediates) {
        return {outputs, state_work, v_buffers_stacked,
                w_num_sens_stacked, w_den_sens_stacked};
    }
    return {outputs, state_work};
}


// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ltc_sensory_fwd", &ltc_sensory_fwd, "LTC sensory forward (CUDA)");
    m.def("ltc_sensory_bwd", &ltc_sensory_bwd, "LTC sensory backward (CUDA)");
    m.def("ltc_ode_unfold_fwd", &ltc_ode_unfold_fwd, "LTC ODE unfold forward (CUDA)");
    m.def("ltc_ode_unfold_bwd", &ltc_ode_unfold_bwd, "LTC ODE unfold backward (CUDA)");
    m.def("ltc_full_forward", &ltc_full_forward, "LTC full forward T-loop (CUDA)");
    m.def("ltc_full_backward", &ltc_full_backward, "LTC full backward T-loop (CUDA)");
}
