#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish_activation(float x) {
    // Mish activation: x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // Numerically stable implementation
    float exp_x = expf(x);
    float softplus = logf(1.0f + exp_x);
    return x * tanhf(softplus);
}

__global__ void fused_bias_double_mish_kernel_opt(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int feature_idx = idx % out_features;
        float val = input[idx] + bias[feature_idx];
        val = mish_activation(val);
        val = mish_activation(val);
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
    // Map 4D tensor parameters to 2D kernel parameters [batch_size, out_features]
    int batch_size = in_batch;
    int out_features = in_width;
    
    // Allocate temporary bias buffer initialized to zero
    // Note: bias is not provided in the interface, allocating zero bias to match kernel signature
    float* bias = nullptr;
    cudaMalloc(&bias, out_features * sizeof(float));
    cudaMemset(bias, 0, out_features * sizeof(float));
    
    const int threads = 256;
    const int blocks = (in_elems + threads - 1) / threads;
    
    fused_bias_double_mish_kernel_opt<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        bias,
        reinterpret_cast<float*>(output),
        batch_size,
        out_features
    );
    
    cudaFree(bias);
}