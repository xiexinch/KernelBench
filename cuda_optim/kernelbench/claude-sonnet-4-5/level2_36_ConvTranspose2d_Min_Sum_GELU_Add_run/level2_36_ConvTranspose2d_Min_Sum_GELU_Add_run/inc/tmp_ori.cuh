#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Note: gelu_activation function is assumed to be defined in external headers 
// (tmp_check.cuh or tmp_use.cuh) and should not be redefined here

__global__ void fused_min_sum_gelu_bias_kernel_opt(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    // Each thread handles one (batch, width) position
    int batch_idx = blockIdx.y;
    int width_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && width_idx < width) {
        float min_val = FLT_MAX;
        
        // Min reduction across channels
        for (int c = 0; c < channels; c++) {
            float sum_val = 0.0f;
            
            // Sum reduction across height
            for (int h = 0; h < height; h++) {
                int idx = batch_idx * (channels * height * width) + 
                         c * (height * width) + 
                         h * width + 
                         width_idx;
                sum_val += input[idx];
            }
            
            min_val = fminf(min_val, sum_val);
        }
        
        // Apply GELU
        float gelu_val = gelu_activation(min_val);
        
        // Add bias
        float result = gelu_val + bias[0];
        
        // Write output
        int out_idx = batch_idx * width + width_idx;
        output[out_idx] = result;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* bias, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Map to kernel parameters
    int batch_size = in_batch;
    int channels = in_channels;
    int height = in_height;
    int width = in_width;
    
    const int threads = 256;
    const int blocks_x = (width + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    fused_min_sum_gelu_bias_kernel_opt<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (const float*)bias,
        (float*)output,
        batch_size,
        channels,
        height,
        width
    );
}