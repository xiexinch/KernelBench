__global__ void fused_relu_bias_kernel_opt(const float* x, const float* bias, float* out, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        float val = x[idx];
        val = val > 0.0f ? val : 0.0f;  // ReLU
        val = val + bias[c];  // Bias addition
        out[idx] = val;
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
    
    T* bias = input + in_elems;
    
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_relu_bias_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, 
        bias, 
        output,
        batch_size, channels, height, width
    );
}