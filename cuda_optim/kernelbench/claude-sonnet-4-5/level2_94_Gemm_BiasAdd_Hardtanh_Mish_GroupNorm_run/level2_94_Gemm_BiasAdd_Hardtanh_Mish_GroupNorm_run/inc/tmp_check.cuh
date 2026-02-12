#include <cuda_runtime.h>
#include <math.h>

__device__ float hardtanh_activation(float x) {
    const float min_val = -1.0f;
    const float max_val = 1.0f;
    return fmaxf(min_val, fminf(max_val, x));
}

__device__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_bias_hardtanh_mish_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * features;
    
    if (idx < total_elements) {
        int feature_idx = idx % features;
        float val = input[idx] + bias[feature_idx];
        val = hardtanh_activation(val);
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
    cudaStream_t stream)
{
    const float* input_f = static_cast<const float*>(input);
    float* output_f = static_cast<float*>(output);
    
    // Assume bias is stored contiguously after input data
    const float* bias_f = input_f + in_elems;
    
    int batch_size = in_batch;
    int features = in_width;
    
    const int threads = 256;
    const int blocks = (in_elems + threads - 1) / threads;
    
    fused_bias_hardtanh_mish_kernel_ori<<<blocks, threads, 0, stream>>>(
        input_f,
        bias_f,
        output_f,
        batch_size,
        features
    );
}