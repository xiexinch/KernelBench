__global__ void group_norm_min_clamp_kernel_ori(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    const int batch_size,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float min_value,
    const float max_value,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < total_groups) {
        int b = idx / num_groups;
        int g = idx % num_groups;
        
        int group_start = b * num_groups * channels_per_group * spatial_size + 
                         g * channels_per_group * spatial_size;
        int group_size = channels_per_group * spatial_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < group_size; i++) {
            sum += input[group_start + i];
        }
        float mean = sum / group_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float diff = input[group_start + i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / group_size;
        float std = sqrtf(variance + eps);
        
        // Normalize, apply gamma/beta, min, and clamp
        for (int c = 0; c < channels_per_group; c++) {
            int channel_idx = g * channels_per_group + c;
            for (int s = 0; s < spatial_size; s++) {
                int pos = group_start + c * spatial_size + s;
                float normalized = (input[pos] - mean) / std;
                float scaled = normalized * gamma[channel_idx] + beta[channel_idx];
                scaled = fminf(scaled, min_value);
                scaled = fmaxf(fminf(scaled, max_value), min_value);
                output[pos] = scaled;
            }
        }
    }
}

__global__ void dropout_kernel_ori(
    const float* input,
    float* output,
    unsigned char* mask,
    const int size,
    const float p,
    const unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        
        if (rand_val < p) {
            output[idx] = 0.0f;
            mask[idx] = 0;
        } else {
            output[idx] = input[idx] / (1.0f - p);
            mask[idx] = 1;
        }
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
    int num_groups = 8;
    int channels = in_channels;
    int channels_per_group = channels / num_groups;
    int spatial_size = in_height * in_width;
    float min_value = 0.0f;
    float max_value = 1.0f;
    float eps = 1e-5f;
    
    int total_groups = batch_size * num_groups;
    const int block_size = 256;
    const int num_blocks = (total_groups + block_size - 1) / block_size;