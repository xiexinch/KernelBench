#include <cuda_runtime.h>
#include <cfloat>

__global__ void max_reduce_channel_kernel_opt(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int channels,
    int spatial_size
) {
    int b = blockIdx.x;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || idx >= spatial_size) return;
    
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        float val = x[(b * channels + c) * spatial_size + idx];
        max_val = fmaxf(max_val, val);
    }
    
    out[b * spatial_size + idx] = max_val;
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    
    int threads = 256;
    int blocks_y = (spatial_size + threads - 1) / threads;
    dim3 blocks(batch_size, blocks_y);
    
    max_reduce_channel_kernel_opt<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (float*)output,
        batch_size,
        channels,
        spatial_size
    );
}