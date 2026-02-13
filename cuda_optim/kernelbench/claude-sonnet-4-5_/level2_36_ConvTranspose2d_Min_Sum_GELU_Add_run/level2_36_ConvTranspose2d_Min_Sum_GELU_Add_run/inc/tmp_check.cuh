#include <cuda_runtime.h>
#include <cfloat>

__device__ __forceinline__ float gelu_activation(float x) {
    const float sqrt2_inv = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * sqrt2_inv));
}

__global__ void fused_min_sum_gelu_bias_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int batch_idx = blockIdx.y;
    int width_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && width_idx < width) {
        float min_val = FLT_MAX;
        
        for (int c = 0; c < channels; c++) {
            float sum_val = 0.0f;
            
            for (int h = 0; h < height; h++) {
                int idx = batch_idx * (channels * height * width) + 
                         c * (height * width) + 
                         h * width + 
                         width_idx;
                sum_val += input[idx];
            }
            
            min_val = fminf(min_val, sum_val);
        }
        
        float gelu_val = gelu_activation(min_val);
        float result = gelu_val + bias[0];
        
        int out_idx = batch_idx * width + width_idx;
        output[out_idx] = result;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* bias, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    int batch_size = in_batch;
    int channels = in_channels;
    int height = in_height;
    int width = in_width;
    
    const int threads = 256;
    const int blocks_x = (width + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    fused_min_sum_gelu_bias_kernel_ori<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (const float*)bias,
        (float*)output,
        batch_size,
        channels,
        height,
        width
    );
}