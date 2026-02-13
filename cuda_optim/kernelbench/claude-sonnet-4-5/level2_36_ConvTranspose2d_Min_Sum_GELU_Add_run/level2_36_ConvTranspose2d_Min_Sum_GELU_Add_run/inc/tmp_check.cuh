#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Extract bias from the end of the input buffer
    // KernelBench convention: bias follows main input data
    const float* bias_ptr = reinterpret_cast<const float*>(input) + in_elems;

    const int threads = 256;
    const int blocks_x = (in_width + threads - 1) / threads;
    dim3 blocks(blocks_x, in_batch);
    
    fused_min_sum_gelu_bias_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        bias_ptr,
        reinterpret_cast<float*>(output),
        in_batch,
        in_channels,
        in_height,
        in_width
    );
}