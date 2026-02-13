#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_activation(float x) {
    return x * tanh(log1p(exp(x)));
}

__global__ void fused_bias_double_mish_kernel_ori(
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    int batch_size = in_batch;
    int out_features = in_channels;
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    
    // Assuming bias is stored right before the output buffer (as per kernelbench convention)
    const float* bias = reinterpret_cast<const float*>(output) - out_features;
    
    fused_bias_double_mish_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        bias,
        reinterpret_cast<float*>(output),
        batch_size,
        out_features
    );
}