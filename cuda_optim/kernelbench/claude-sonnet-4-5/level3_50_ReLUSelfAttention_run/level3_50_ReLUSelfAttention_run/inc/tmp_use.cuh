#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_qk_mask_relu_kernel_opt(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ att,
    const float scale,
    const int B, const int H, const int T, const int hs
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int t_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t_row >= T) return;
    
    const int qk_offset = b * H * T * hs + h * T * hs;
    const int att_offset = b * H * T * T + h * T * T;
    
    for (int t_col = 0; t_col < T; ++t_col) {
        float sum = 0.0f;
        
        // Compute dot product between q[t_row] and k[t_col]
        for (int i = 0; i < hs; ++i) {
            sum += q[qk_offset + t_row * hs + i] * k[qk_offset + t_col * hs + i];
        }
        
        // Scale
        sum *= scale;
        
        // Apply causal mask and ReLU
        if (t_col > t_row) {
            sum = 0.0f;  // Masked position
        } else {
            sum = fmaxf(sum, 0.0f);  // ReLU
        }
        
        att[att_offset + t_row * T + t_col] = sum;
    }
}

__global__ void attention_matmul_v_kernel_opt(
    const float* __restrict__ att,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int B, const int H, const int T, const int hs
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t >= T) return;
    
    const int att_offset = b * H * T * T + h * T * T;
    const int v_offset = b * H * T * hs + h * T * hs;
    const int out_offset = b * H * T * hs + h * T * hs;
    
    for (int i = 0; i < hs; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < T; ++j) {
            sum += att[att_offset + t * T + j] * v[v_offset + j * hs + i];
        }
        out[out_offset + t * hs + i] = sum;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Map input parameters to attention dimensions
    const int B = in_batch;      // batch size
    const int H = in_height;     // number of heads
    const int T = in_width;      // sequence length
    const int hs = in_channels;  // head size

    const float scale = 1.0f / sqrtf(static_cast<float>(hs));

    // Allocate temporary attention matrix [B, H, T, T]
    float* att = nullptr;
    size_t att_size = static_cast<size_t>(B) * H * T * T * sizeof(float);
    cudaMalloc(&att, att_size);

    const int threads = 256;
    const int blocks_t = (T + threads - 1) / threads;
    dim3 blocks(blocks_t, H, B);

    // Step 1: Compute QK^T with mask and ReLU
    // q is at input[0:B*H*T*hs]
    // k is at input[B*H*T*hs:2*B*H*T*hs]
    const float* q_ptr = reinterpret_cast<const float*>(input);
    const float* k_ptr = reinterpret_cast<const float*>(input) + static_cast<size_t>(B) * H * T * hs;
    
    fused_qk_mask_relu_kernel_opt<<<blocks, threads, 0, stream>>>(
        q_ptr,
        k_ptr,
        att,
        scale, B, H, T, hs
    );

    // Step 2: Compute att * V
    // v is at input[2*B*H*T*hs:3*B*H*T*hs]
    const float* v_ptr = reinterpret_cast<const float*>(input) + 2 * static_cast<size_t>(B) * H * T * hs;
    float* out_ptr = reinterpret_cast<float*>(output);
    
    attention_matmul_v_kernel_opt<<<blocks, threads, 0, stream>>>(
        att,
        v_ptr,
        out_ptr,
        B, H, T, hs
    );

    cudaFree(att);
}