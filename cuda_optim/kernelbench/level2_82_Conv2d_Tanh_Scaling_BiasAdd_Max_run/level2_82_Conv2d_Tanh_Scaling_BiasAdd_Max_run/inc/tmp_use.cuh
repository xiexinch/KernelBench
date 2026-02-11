__global__ void fused_tanh_scale_bias_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        float val = input[idx];
        val = tanhf(val);
        val = val * scaling_factor;
        val = val + bias[c];
        output[idx] = val;
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
    int batch_size = in_batch;
    int channels = in_channels;
    int height = in_height;
    int width = in_width;
    int total_size = batch_size * channels * height * width;
    float scaling_factor = 2.0;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        input,
        output,
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );
}