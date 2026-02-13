#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_scale_add_clamp_kernel_ori(
    const float* input,
    float* output,
    const float scale_factor,
    const float clamp_min,
    const float clamp_max,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scale_factor;
        val = val + val;  // x + x
        val = fmaxf(clamp_min, fminf(val, clamp_max));
        output[idx] = val;
    }
}

__global__ void logsumexp_kernel_ori(
    const float* input,
    float* output,
    const int batch_size,
    const int hidden_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        const float* row = input + batch_idx * hidden_size;
        
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < hidden_size; ++i) {
            max_val = fmaxf(max_val, row[i]);
        }
        
        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum_exp += expf(row[i] - max_val);
        }
        
        // logsumexp = max + log(sum_exp)
        output[batch_idx] = max_val + logf(sum_exp);
    }
}

__global__ void fused_mish_mul_kernel_ori(
    const float* input,
    float* output,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float mish_val = mish_activation(val);
        output[idx] = val * mish_val;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    const int block_size = 256;
    
    // Handle fused_scale_add_clamp: same input and output shape
    if (in_batch == out_batch && in_height == out_height && 
        in_channels == out_channels && in_width == out_width && 
        in_elems == out_elems) {
        // Check if it's likely the first kernel (full feature dimension)
        if (in_elems > in_batch) {
            float scale_factor = 1.0f;
            float clamp_min = -10.0f;
            float clamp_max = 10.0f;
            int num_blocks = (in_elems + block_size - 1) / block_size;
            fused_scale_add_clamp_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                scale_factor, clamp_min, clamp_max, in_elems);
        } else {
            // Handle fused_mish_mul: same shape but typically scalar per batch
            int num_blocks = (in_elems + block_size - 1) / block_size;
            fused_mish_mul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                in_elems);
        }
    }
    // Handle logsumexp: reduction from (batch, hidden) to (batch,)
    else if (out_batch == in_batch && out_elems == in_batch && 
             out_height == 1 && out_width == 1 && out_channels == 1) {
        int hidden_size = in_elems / in_batch;
        int num_blocks = (in_batch + block_size - 1) / block_size;
        logsumexp_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            in_batch, hidden_size);
    }
}