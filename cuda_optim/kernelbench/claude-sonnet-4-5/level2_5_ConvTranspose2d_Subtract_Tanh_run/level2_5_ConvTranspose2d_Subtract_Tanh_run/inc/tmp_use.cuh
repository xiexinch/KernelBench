__global__ void fused_bias_tanh_kernel_opt(const float* input, const float* bias, 
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    const int block_size = 256;
    int total_size = in_batch * in_channels * in_height * in_width;
    int num_blocks = (total_size + block_size - 1) / block_size;

    // Assuming bias is stored right after input in memory (as per kernelbench convention)
    // In this specific kernel, bias is a separate array of size 'in_channels'
    // For the purpose of this benchmark wrapper, we assume bias resides at input + in_elems
    T* bias = input + in_elems;

    fused_bias_tanh_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, bias, output, in_batch, in_channels, in_height, in_width
    );
}