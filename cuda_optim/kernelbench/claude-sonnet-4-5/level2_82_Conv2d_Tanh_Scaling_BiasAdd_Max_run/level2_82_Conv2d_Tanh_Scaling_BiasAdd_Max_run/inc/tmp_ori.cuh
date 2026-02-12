#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_tanh_scale_bias_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        float val = input[idx];
        val = tanhf(val);
        val = val * scaling_factor;
        if (bias != nullptr) {
            val = val + bias[c];
        }
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
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    const float* input_f = (const float*)input;
    float* output_f = (float*)output;
    
    const float scaling_factor = 2.0f;
    const float* bias = nullptr;
    
    fused_tanh_scale_bias_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input_f,
        bias,
        output_f,
        scaling_factor,
        in_batch,
        in_channels,
        in_height,
        in_width
    );
}