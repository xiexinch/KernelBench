__global__ void group_norm_scale_kernel_ori(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* scale,
    float* output,
    int batch_size,
    int num_channels,
    int spatial_size,
    int num_groups,
    int channels_per_group,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < batch_size * num_channels * spatial_size) {
        int s = idx % spatial_size;
        int c = (idx / spatial_size) % num_channels;
        int b = idx / (num_channels * spatial_size);
        int g = c / channels_per_group;
        
        // Calculate mean and variance for this group
        float sum = 0.0f;
        float sq_sum = 0.0f;
        int group_size = channels_per_group * spatial_size;
        int group_start_c = g * channels_per_group;
        
        for (int gc = 0; gc < channels_per_group; gc++) {
            for (int gs = 0; gs < spatial_size; gs++) {
                int gidx = b * num_channels * spatial_size + (group_start_c + gc) * spatial_size + gs;
                float val = input[gidx];
                sum += val;
                sq_sum += val * val;
            }
        }
        
        float mean = sum / group_size;
        float var = sq_sum / group_size - mean * mean;
        float std_inv = rsqrtf(var + eps);
        
        // Normalize and apply affine transformation
        float normalized = (input[idx] - mean) * std_inv;
        float affine = normalized * gamma[c] + beta[c];
        
        // Apply scale
        output[idx] = affine * scale[c];
    }
}

__global__ void maxpool_clamp_kernel_ori(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (channels * out_height * out_width);
        
        float max_val = -1e38f;
        
        int h_start = oh * kernel_size;
        int w_start = ow * kernel_size;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h = h_start + kh;
                int w = w_start + kw;
                
                if (h < in_height && w < in_width) {
                    int in_idx = b * channels * in_height * in_width + 
                                c * in_height * in_width + 
                                h * in_width + w;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        // Apply clamp
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
        output[idx] = max_val;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int