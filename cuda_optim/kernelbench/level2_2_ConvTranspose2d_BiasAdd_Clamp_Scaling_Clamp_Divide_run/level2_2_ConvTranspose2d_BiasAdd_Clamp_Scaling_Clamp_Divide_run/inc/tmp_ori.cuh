__global__ void fused_bias_clamp_scale_kernel_opt(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        int bias_idx = c;
        
        float val = input[idx] + bias[bias_idx];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val / scaling_factor;
        
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
    float scaling_factor = 2.0;
    
    int total_elements = in_elems;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    T* bias = nullptr;
    
    fused_bias_clamp_scale_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        bias,
        output,
        batch_size,
        channels,
        height,
        width,
        scaling_factor
    );
}