__global__ void fused_bias_tanh_kernel_ori(const float* input, const float* bias, 
                                        float* output, int batch_size, int channels, 
                                        int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        float val = input[idx] - bias[c];
        output[idx] = tanhf(val);
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
    int channels = in_channels;
    int height = in_height;
    int width = in_width;
    
    int total_size = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    T* bias = input + in_elems;
    
    fused_bias_tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, 
        bias, 
        output, 
        batch_size, channels, height, width
    );
}