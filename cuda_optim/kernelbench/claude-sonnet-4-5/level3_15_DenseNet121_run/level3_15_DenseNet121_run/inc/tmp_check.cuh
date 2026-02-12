__global__ void batchnorm_relu_kernel_ori(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int batch_size,
    int channels,
    int spatial_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int spatial_idx = idx % spatial_size;
        int c = (idx / spatial_size) % channels;
        int b = idx / (spatial_size * channels);
        
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float bi = bias[c];
        
        float normalized = (x[idx] - mean) / sqrtf(var + eps);
        float scaled = normalized * w + bi;
        out[idx] = fmaxf(scaled, 0.0f);  // ReLU
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream,
    T* weight, T* bias, T* running_mean, T* running_var, float eps)
{
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    int total_size = batch_size * channels * spatial_size;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    batchnorm_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        output,
        batch_size,
        channels,
        spatial_size,
        eps
    );
}