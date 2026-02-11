__global__ void fused_swish_scale_kernel_ori(const float* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x * scale;
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
    int size = in_elems;
    float scale = 2.0;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_swish_scale_kernel_ori<<<num_blocks, block_size, 0, stream>>>(input, output, scale, size);
}