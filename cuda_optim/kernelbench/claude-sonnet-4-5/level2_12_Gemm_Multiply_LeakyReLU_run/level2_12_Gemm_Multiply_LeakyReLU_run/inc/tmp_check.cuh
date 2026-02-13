__global__ void fused_multiply_leaky_relu_kernel_ori(
    const float* input, 
    float* output, 
    float multiplier, 
    float negative_slope, 
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * multiplier;
        output[idx] = val > 0.0f ? val : val * negative_slope;
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
    float multiplier = 2.0f;
    float negative_slope = 0.1f;
    int size = in_elems;

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_multiply_leaky_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, output, multiplier, negative_slope, size);
}