#include <cuda_runtime.h>
#include <math.h>

// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// Guarded to avoid redefinition if already provided by environment headers
#ifndef MISH_ACTIVATION_DEFINED
#define MISH_ACTIVATION_DEFINED
__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}
#endif

__global__ void fused_mish_add_hardtanh_scale_kernel_opt(
    const float* input,
    float* output,
    const float add_value,
    const float scale,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Apply Mish
        val = mish_activation(val);
        // Add value
        val = val + add_value;
        // Apply Hardtanh (clamp between -1 and 1)
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        // Scale
        val = val * scale;
        output[idx] = val;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    const float add_value = 0.5f;
    const float scale = 2.0f;
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_mish_add_hardtanh_scale_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        add_value,
        scale,
        in_elems
    );
}