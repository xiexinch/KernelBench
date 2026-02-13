#include <cuda_runtime.h>
#include <cmath>

__device__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__device__ float tanh_activation(float x) {
    return tanhf(x);
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

__device__ float hardtanh(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

__global__ void fused_activations_kernel_opt(const float* input, const float* add_value, 
                                         float* output, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * features;
    
    if (idx < total_size) {
        int feature_idx = idx % features;
        float val = input[idx] + add_value[feature_idx];
        val = swish(val);
        val = tanh_activation(val);
        val = gelu(val);
        val = hardtanh(val, -1.0f, 1.0f);
        output[idx] = val;
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
    const float* add_value = reinterpret_cast<const float*>(input) + in_elems;
    int batch_size = in_batch;
    int features = in_width;

    const int threads = 256;
    const int blocks = (in_elems + threads - 1) / threads;

    fused_activations_kernel_opt<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        add_value,
        reinterpret_cast<float*>(output),
        batch_size,
        features
    );
}