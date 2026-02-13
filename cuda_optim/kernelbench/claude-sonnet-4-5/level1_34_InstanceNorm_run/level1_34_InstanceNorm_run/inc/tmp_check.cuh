__global__ void instance_norm_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps
) {
    // Each block processes one (batch, channel) pair
    int bc_idx = blockIdx.x;
    int batch_idx = bc_idx / num_features;
    int channel_idx = bc_idx % num_features;
    
    if (batch_idx >= batch_size || channel_idx >= num_features) return;
    
    // Calculate offset for this batch and channel
    int offset = (batch_idx * num_features + channel_idx) * spatial_size;
    
    // Compute mean using reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    
    // Reduce within block using shared memory
    __shared__ float shared_data[256];
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / spatial_size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    
    shared_data[threadIdx.x] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float variance = shared_data[0] / spatial_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and apply affine transformation
    float w = weight[channel_idx];
    float b = bias[channel_idx];
    
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * w + b;
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
    int num_features = in_channels;
    int spatial_size = in_height * in_width;
    float eps = 1e-5f;

    const int block_size = 256;
    const int num_blocks = batch_size * num_features;

    instance_norm_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        nullptr,  // weight not used in this test interface
        nullptr,  // bias not used in this test interface
        batch_size,
        num_features,
        spatial_size,
        eps
    );
}