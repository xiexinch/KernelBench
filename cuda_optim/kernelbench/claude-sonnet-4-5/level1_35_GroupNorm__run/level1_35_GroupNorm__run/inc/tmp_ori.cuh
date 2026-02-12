__global__ void group_norm_forward_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int num_channels,
    int num_groups,
    int spatial_size,
    float eps) {
    
    int channels_per_group = num_channels / num_groups;
    int group_size = channels_per_group * spatial_size;
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    int base_idx = batch_idx * num_channels * spatial_size + group_idx * channels_per_group * spatial_size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += input[base_idx + i];
    }
    
    // Reduce sum across threads in block
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / group_size;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }
    
    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float variance = shared_sum[0] / group_size;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();
    
    // Normalize and apply affine transformation
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int channel_offset = i / spatial_size;
        int channel_idx = group_idx * channels_per_group + channel_offset;
        float normalized = (input[base_idx + i] - mean) * inv_std;
        output[base_idx + i] = normalized * gamma[channel_idx] + beta[channel_idx];
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream,
    T* gamma,
    T* beta,
    int num_groups,
    float eps)
{
    int batch_size = in_batch;
    int num_channels = in_channels;
    int spatial_size = in_height * in_width;
    
    dim3 blocks(batch_size, num_groups);
    int threads = 256;
    
    group_norm_forward_kernel_opt<<<blocks, threads, 0, stream>>>(
        input,
        gamma,
        beta,
        output,
        batch_size,
        num_channels,
        num_groups,
        spatial_size,
        eps
    );
}