#include <cuda_runtime.h>
#include <math.h>

__global__ void groupnorm_swish_kernel_opt(
    const float* x, 
    const float* gamma, 
    const float* beta,
    float* out, 
    int batch_size,
    int channels,
    int group_size,
    int num_groups,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < total_groups) {
        int b = idx / num_groups;
        int g = idx % num_groups;
        
        // Calculate mean
        float sum = 0.0f;
        int start_c = g * group_size;
        int end_c = start_c + group_size;
        
        for (int c = start_c; c < end_c; c++) {
            sum += x[b * channels + c];
        }
        float mean = sum / group_size;
        
        // Calculate variance
        float var_sum = 0.0f;
        for (int c = start_c; c < end_c; c++) {
            float diff = x[b * channels + c] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / group_size;
        float std = sqrtf(var + eps);
        
        // Normalize, scale, shift, and apply swish
        for (int c = start_c; c < end_c; c++) {
            float normalized = (x[b * channels + c] - mean) / std;
            float scaled = normalized * gamma[c] + beta[c];
            float sigmoid_val = 1.0f / (1.0f + expf(-scaled));
            out[b * channels + c] = scaled * sigmoid_val;
        }
    }
}

__global__ void multiply_swish_kernel_v2_opt(
    const float* x,
    const float* weight,
    float* out,
    int batch_size,
    int channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels;
    
    if (idx < total_size) {
        int c = idx % channels;
        float weighted = x[idx] * weight[c];
        float sigmoid_val = 1.0f / (1.0f + expf(-weighted));
        out[idx] = weighted * sigmoid_val;
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
    // Determine which kernel to launch based on dimensions
    // For groupnorm_swish: expects 2D layout [batch, channels], no spatial dims
    // So we assume in_height == in_width == 1
    if (in_height == 1 && in_width == 1 && out_height == 1 && out_width == 1) {
        int batch_size = in_batch;
        int channels = in_channels;
        int num_groups = 32; // default from example
        int group_size = channels / num_groups;
        float eps = 1e-5f;

        const int block_size = 256;
        const int num_blocks = (batch_size * num_groups + block_size - 1) / block_size;

        groupnorm_swish_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const float*>(input + in_elems),      // gamma after input
            reinterpret_cast<const float*>(input + in_elems * 2),  // beta after gamma
            reinterpret_cast<float*>(output),
            batch_size,
            channels,
            group_size,
            num_groups,
            eps
        );
    } else {
        // Otherwise assume multiply_swish path
        int batch_size = in_batch * in_height * in_width;
        int channels = in_channels;

        const int block_size = 256;
        const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

        multiply_swish_kernel_v2_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const float*>(input + in_elems), // weight after input
            reinterpret_cast<float*>(output),
            batch_size,
            channels
        );
    }
}