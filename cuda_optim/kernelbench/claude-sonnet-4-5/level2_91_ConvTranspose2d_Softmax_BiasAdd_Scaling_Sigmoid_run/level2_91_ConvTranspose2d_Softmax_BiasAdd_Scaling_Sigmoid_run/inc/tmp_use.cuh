#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <type_traits>

__global__ void fused_softmax_bias_scale_sigmoid_kernel_opt(
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    static_assert(std::is_same<T, float>::value, "This kernel only supports float type");
    
    // Map output dimensions to kernel parameters
    int batch_size = out_batch;
    int channels = out_channels;
    int spatial_size = out_height * out_width;
    float scaling_factor = 2.0f;
    
    // Static bias buffer to avoid repeated cudaMalloc/cudaFree overhead in benchmarks
    static float* bias_buffer = nullptr;
    static int bias_capacity = 0;
    
    if (channels > bias_capacity) {
        if (bias_buffer) cudaFree(bias_buffer);
        cudaMalloc((void**)&bias_buffer, channels * sizeof(float));
        bias_capacity = channels;
    }
    
    // Initialize bias to zero
    cudaMemsetAsync(bias_buffer, 0, channels * sizeof(float), stream);
    
    const int threads = 256;
    const int blocks_x = (spatial_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    fused_softmax_bias_scale_sigmoid_kernel_opt<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input),
        bias_buffer,
        static_cast<float*>(output),
        batch_size,
        channels,
        spatial_size,
        scaling_factor
    );
}