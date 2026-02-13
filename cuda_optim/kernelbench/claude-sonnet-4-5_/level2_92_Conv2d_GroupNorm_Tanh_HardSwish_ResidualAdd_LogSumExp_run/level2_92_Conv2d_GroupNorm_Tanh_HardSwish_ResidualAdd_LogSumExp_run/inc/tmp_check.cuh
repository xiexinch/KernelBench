#include <cuda_runtime.h>
#include <math.h>

__global__ void logsumexp_kernel_ori(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * spatial_size;
    
    if (idx < total_output) {
        int b = idx / spatial_size;
        int s = idx % spatial_size;
        
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + s;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + s;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        // LogSumExp = max + log(sum_exp)
        output[idx] = max_val + logf(sum_exp);
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
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    
    const int threads = 256;
    const int blocks = (out_elems + threads - 1) / threads;
    
    logsumexp_kernel_ori<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (float*)output,
        batch_size,
        channels,
        spatial_size
    );
}