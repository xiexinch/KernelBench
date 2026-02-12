#include <cuda_runtime.h>
#include <math.h>

// Utility kernel for initialization
__global__ void fill_constant_kernel_opt(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// Original kernel 1: Swish activation with bias addition
__global__ void swish_bias_kernel_opt(const float* x, const float* bias, float* out, int size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % out_features;
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = sigmoid_val * val + bias[bias_idx];
    }
}

// Original kernel 2: GroupNorm statistics computation
__global__ void group_norm_stats_kernel_opt(const float* x, float* mean, float* var, 
                                         int batch_size, int num_groups, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < total_groups) {
        int offset = idx * group_size;
        float sum = 0.0f;
        float sq_sum = 0.0f;
        
        for (int i = 0; i < group_size; i++) {
            float val = x[offset + i];
            sum += val;
            sq_sum += val * val;
        }
        
        mean[idx] = sum / group_size;
        var[idx] = sq_sum / group_size - mean[idx] * mean[idx];
    }
}

// Original kernel 3: GroupNorm forward pass
__global__ void group_norm_forward_kernel_opt(const float* x, const float* mean, const float* var,
                                          const float* weight, const float* bias, float* out,
                                          int batch_size, int num_groups, int group_size, 
                                          int channels, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels;
    
    if (idx < total_size) {
        int batch_idx = idx / channels;
        int channel_idx = idx % channels;
        int group_idx = channel_idx / (channels / num_groups);
        int stat_idx = batch_idx * num_groups + group_idx;
        
        float m = mean[stat_idx];
        float v = var[stat_idx];
        float std_inv = rsqrtf(v + eps);
        
        float normalized = (x[idx] - m) * std_inv;
        out[idx] = normalized * weight[channel_idx] + bias[channel_idx];
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
    // Cast to float pointers (kernels are float-specific)
    float* x = reinterpret_cast<float*>(input);
    float* out = reinterpret_cast<float*>(output);
    
    const int block_size = 256;
    
    // Calculate dimensions
    int batch_size = in_batch;
    int channels = in_channels * in_height * in_width;
    int total_size = batch_size * channels;
    
    // Swish+Bias parameters
    int out_features = channels;  // Assuming last dimension is features
    
    // GroupNorm parameters
    int num_groups = 32;
    if (channels % num_groups != 0) num_groups = 16;
    if (channels % num_groups != 0) num_groups = 8;
    if (channels % num_groups != 0) num_groups = 1;
    int group_size = channels / num_groups;
    int total_groups = batch_size * num_groups;
    float eps = 1e-5f;
    
    // Allocate intermediate buffers
    float *swish_bias, *mean, *var, *gn_weight, *gn_bias;
    cudaMalloc(&swish_bias, total_size * sizeof(float));
    cudaMalloc(&mean, total_groups * sizeof(float));
    cudaMalloc(&var, total_groups * sizeof(float));
    cudaMalloc(&gn_weight, channels * sizeof(float));
    cudaMalloc(&gn_bias, channels * sizeof(float));
    
    // Initialize parameters: swish_bias buffer, GN weight to 1.0, GN bias to 0.0
    int init_blocks = (channels + block_size - 1) / block_size;
    fill_constant_kernel_opt<<<init_blocks, block_size, 0, stream>>>(swish_bias, 0.0f, total_size);
    fill_constant_kernel_opt<<<init_blocks, block_size, 0, stream>>>(gn_weight, 1.0f, channels);
    cudaMemsetAsync(gn_bias, 0, channels * sizeof(float), stream);
    
    // Step 1: Swish + Bias
    int num_blocks_swish = (total_size + block_size - 1) / block_size;
    swish_bias_kernel_opt<<<num_blocks_swish, block_size, 0, stream>>>(
        x, swish_bias, out, total_size, out_features
    );
    
    // Step 2: GroupNorm statistics
    int num_blocks_stats = (total_groups + block_size - 1) / block_size;
    group_norm_stats_kernel_opt<<<num_blocks_stats, block_size, 0, stream>>>(
        out, mean, var, batch_size, num_groups, group_size
    );
    
    // Step 3: GroupNorm forward (in-place on output)
    int num_blocks_forward = (total_size + block_size - 1) / block_size;
    group_norm_forward_kernel_opt<<<num_blocks_forward, block_size, 0, stream>>>(
        out, mean, var, gn_weight, gn_bias, out,
        batch_size, num_groups, group_size, channels, eps
    );
    
    // Cleanup
    cudaFree(swish_bias);
    cudaFree(mean);
    cudaFree(var);
    cudaFree(gn_weight);
    cudaFree(gn_bias);
}

// Explicit instantiation for float
template void test_tmp_kernel_opt<float>(
    float* input, float* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
);