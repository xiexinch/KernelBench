__global__ void fused_groupnorm_tanh_hardswish_residual_kernel_opt(
    const float* x_conv,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    int groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int b = idx / (channels * spatial_size);
        int c = (idx / spatial_size) % channels;
        int s = idx % spatial_size;
        
        int group_id = c / (channels / groups);
        int channels_per_group = channels / groups;
        int group_size = channels_per_group * spatial_size;
        
        // Compute mean and variance for the group
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int base_idx = b * channels * spatial_size + group_id * channels_per_group * spatial_size;
        
        for (int i = 0; i < group_size; i++) {
            float val = x_conv[base_idx + i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / group_size;
        float variance = sum_sq / group_size - mean * mean;
        float std = sqrtf(variance + eps);
        
        // Normalize
        float x_val = x_conv[idx];
        float normalized = (x_val - mean) / std;
        
        // Apply gamma and beta
        float x_norm = normalized * gamma[c] + beta[c];
        
        // Apply Tanh
        float x_tanh = tanhf(x_norm);
        
        // Apply HardSwish: x * relu6(x + 3) / 6
        float x_plus_3 = x_tanh + 3.0f;
        float relu6_val = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
        float x_hard_swish = x_tanh * relu6_val / 6.0f;
        
        // Add residual
        output[idx] = x_val + x_hard_swish;
    }
}

__global__ void logsumexp_kernel_opt(
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int