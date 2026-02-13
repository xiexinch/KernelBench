#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_maxpool_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features / 2) return;
    
    // Compute two consecutive outputs from linear layer
    float val1 = 0.0f;
    float val2 = 0.0f;
    
    int out_idx1 = out_idx * 2;
    int out_idx2 = out_idx * 2 + 1;
    
    // Matrix multiplication
    for (int i = 0; i < in_features; i++) {
        float in_val = input[batch_idx * in_features + i];
        val1 += in_val * weight[out_idx1 * in_features + i];
        val2 += in_val * weight[out_idx2 * in_features + i];
    }
    
    // Add bias
    val1 += bias[out_idx1];
    val2 += bias[out_idx2];
    
    // Max pooling
    float max_val = fmaxf(val1, val2);
    
    output[batch_idx * (out_features / 2) + out_idx] = max_val;
}

__global__ void fused_sum_scale_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int features,
    float scale_factor
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < features; i++) {
        sum += input[batch_idx * features + i];
    }
    
    output[batch_idx] = sum * scale_factor;
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Cast to float since original kernels operate on float
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Map dimensions to original kernel parameters
    int batch_size = in_batch;
    int in_features = in_channels * in_height * in_width;
    int out_features_after_maxpool = out_channels * out_height * out_width;
    int out_features_linear = out_features_after_maxpool * 2;
    
    // Assume input layout: [input_data, weight, bias]
    // input_data: batch_size * in_features
    // weight: out_features_linear * in_features  
    // bias: out_features_linear
    float* input_data = input_f;
    float* weight = input_f + batch_size * in_features;
    float* bias = weight + out_features_linear * in_features;
    
    // Allocate intermediate buffer for maxpool output
    float* intermediate;
    cudaMallocAsync(&intermediate, batch_size * out_features_after_maxpool * sizeof(float), stream);
    
    // Launch fused linear + maxpool kernel
    const int threads1 = 256;
    const int blocks_x = (out_features_after_maxpool + threads1 - 1) / threads1;
    dim3 blocks(blocks_x, batch_size);
    
    fused_linear_maxpool_kernel_opt<<<blocks, threads1, 0, stream>>>(
        input_data, weight, bias, intermediate,
        batch_size, in_features, out_features_linear
    );
    
    // Launch fused sum + scale kernel
    const int threads2 = 256;
    const int blocks2 = (batch_size + threads2 - 1) / threads2;
    float scale_factor = 0.5f;  // Original scale factor from model
    
    fused_sum_scale_kernel_opt<<<blocks2, threads2, 0, stream>>>(
        intermediate, output_f, batch_size, out_features_after_maxpool, scale_factor
    );
    
    cudaFreeAsync(intermediate, stream);
}