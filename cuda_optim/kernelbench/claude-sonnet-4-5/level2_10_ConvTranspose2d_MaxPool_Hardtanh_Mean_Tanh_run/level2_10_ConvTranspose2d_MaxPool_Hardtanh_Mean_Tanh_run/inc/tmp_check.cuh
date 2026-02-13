#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void fused_maxpool_hardtanh_mean_tanh_kernel_ori(
    const float* input, 
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int pool_size,
    int pool_stride,
    float hardtanh_min,
    float hardtanh_max
) {
    int bc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = bc_idx / channels;
    int c = bc_idx % channels;
    
    if (b >= batch_size || c >= channels) return;
    
    int out_height = (in_height - pool_size) / pool_stride + 1;
    int out_width = (in_width - pool_size) / pool_stride + 1;
    
    float sum = 0.0f;
    int count = 0;
    
    // Perform maxpool and hardtanh
    for (int oh = 0; oh < out_height; oh++) {
        for (int ow = 0; ow < out_width; ow++) {
            float max_val = -FLT_MAX;
            
            // Max pooling window
            for (int kh = 0; kh < pool_size; kh++) {
                for (int kw = 0; kw < pool_size; kw++) {
                    int ih = oh * pool_stride + kh;
                    int iw = ow * pool_stride + kw;
                    
                    if (ih < in_height && iw < in_width) {
                        int input_idx = b * channels * in_height * in_width + 
                                       c * in_height * in_width + 
                                       ih * in_width + iw;
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
                }
            }
            
            // Apply hardtanh
            float clamped_val = fminf(fmaxf(max_val, hardtanh_min), hardtanh_max);
            sum += clamped_val;
            count++;
        }
    }
    
    // Compute mean and apply tanh
    float mean_val = sum / count;
    float tanh_val = tanhf(mean_val);
    
    // Write output
    int output_idx = b * channels + c;
    output[output_idx] = tanh_val;
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int batch_size = in_batch;
    int channels = in_channels;
    int pool_size = 2; // inferred from example usage
    int pool_stride = 2; // inferred from example usage
    float hardtanh_min = -1.0f; // inferred from example usage
    float hardtanh_max = 1.0f; // inferred from example usage

    int total_threads = batch_size * channels;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_maxpool_hardtanh_mean_tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        channels,
        in_height,
        in_width,
        pool_size,
        pool_stride,
        hardtanh_min,
        hardtanh_max
    );
}