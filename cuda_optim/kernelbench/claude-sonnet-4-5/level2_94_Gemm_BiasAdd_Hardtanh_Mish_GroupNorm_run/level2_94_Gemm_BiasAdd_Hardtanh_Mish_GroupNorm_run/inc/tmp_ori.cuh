#include <cuda_runtime.h>
#include <cfloat>

#ifndef HARDTANH_ACTIVATION_DEFINED
#define HARDTANH_ACTIVATION_DEFINED
__device__ inline float hardtanh_activation(float x) {
    return fmaxf(-2.0f, fminf(2.0f, x));
}
#endif

#ifndef MISH_ACTIVATION_DEFINED
#define MISH_ACTIVATION_DEFINED
__device__ inline float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}
#endif

__global__ void fused_bias_hardtanh_mish_kernel_opt(
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    int batch_size = in_batch;
    int features = in_elems / in_batch;
    const int threads = 256;
    const int blocks = (in_elems + threads - 1) / threads;

    const float* bias = reinterpret_cast<const float*>(input) + in_elems;

    fused_bias_hardtanh_mish_kernel_opt<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        bias,
        reinterpret_cast<float*>(output),
        batch_size,
        features
    );
}