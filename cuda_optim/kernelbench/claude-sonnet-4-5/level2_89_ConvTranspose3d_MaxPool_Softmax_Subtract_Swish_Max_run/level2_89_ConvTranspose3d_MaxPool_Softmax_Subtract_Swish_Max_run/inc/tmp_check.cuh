#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_softmax_subtract_swish_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ subtract_vals,
    float* __restrict__ output,
    int batch_size, int channels, int spatial_size) {
    
    int batch_idx = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= spatial_size) return;
    
    int base_offset = batch_idx * channels * spatial_size + spatial_idx;
    
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        float val = input[base_offset + c * spatial_size];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        float val = expf(input[base_offset + c * spatial_size] - max_val);
        sum += val;
    }
    
    // Softmax + subtract + swish
    for (int c = 0; c < channels; c++) {
        float softmax_val = expf(input[base_offset + c * spatial_size] - max_val) / sum;
        float subtract_val = softmax_val - subtract_vals[c];
        float sigmoid_val = 1.0f / (1.0f + expf(-subtract_val));
        output[base_offset + c * spatial_size] = sigmoid_val * subtract_val;
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
    const int blocks_x = (spatial_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);

    fused_softmax_subtract_swish_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output), // Note: original subtract_vals is passed via output pointer in this benchmark setup
        reinterpret_cast<float*>(output),
        batch_size, channels, spatial_size
    );
}