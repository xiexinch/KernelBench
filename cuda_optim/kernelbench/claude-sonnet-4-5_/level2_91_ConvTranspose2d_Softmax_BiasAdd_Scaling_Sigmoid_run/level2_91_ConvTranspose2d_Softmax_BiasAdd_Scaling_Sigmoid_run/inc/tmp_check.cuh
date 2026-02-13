#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_softmax_bias_scale_sigmoid_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float scaling_factor
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (spatial_idx >= spatial_size || batch_idx >= batch_size) return;
    
    int base_idx = batch_idx * channels * spatial_size + spatial_idx;
    
    // Step 1: Find max for numerical stability in softmax
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        int idx = base_idx + c * spatial_size;
        max_val = fmaxf(max_val, input[idx]);
    }
    
    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        int idx = base_idx + c * spatial_size;
        sum_exp += expf(input[idx] - max_val);
    }
    
    // Step 3: Compute softmax, add bias, scale, and apply sigmoid
    for (int c = 0; c < channels; c++) {
        int idx = base_idx + c * spatial_size;
        float softmax_val = expf(input[idx] - max_val) / sum_exp;
        float biased_val = softmax_val + bias[c];
        float scaled_val = biased_val * scaling_factor;
        output[idx] = 1.0f / (1.0f + expf(-scaled_val));
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
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    float scaling_factor = 2.0f;
    
    float* bias = nullptr;
    cudaMalloc(&bias, channels * sizeof(float));
    cudaMemset(bias, 0, channels * sizeof(float));
    
    const int threads = 256;
    int blocks_x = (spatial_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    fused_softmax_bias_scale_sigmoid_kernel_ori<<<blocks, threads, 0, stream>>>(
        input_f, bias, output_f, batch_size, channels, spatial_size, scaling_factor
    );
    
    cudaFree(bias);
}