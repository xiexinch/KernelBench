__global__ void swish_bias_kernel_opt(const float* x, const float* bias, float* out, int size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % out_features;
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = sigmoid_val * val + bias[bias_idx];
    }
}

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
    cudaStream_t stream)
{
    int size = in_elems;
    int out_features = out_channels;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    T* bias = nullptr;
    cudaMalloc(&bias, out_features * sizeof(T));
    cudaMemset(bias, 0, out_features * sizeof(T));
    
    swish_bias_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, 
        bias, 
        output, 
        size, 
        out_features
    );
    
    cudaFree(bias);
}