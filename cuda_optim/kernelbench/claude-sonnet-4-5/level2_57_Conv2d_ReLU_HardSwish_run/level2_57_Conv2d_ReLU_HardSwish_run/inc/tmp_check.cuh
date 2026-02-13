__global__ void fused_relu_hardswish_kernel_ori(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Apply ReLU
        x = fmaxf(0.0f, x);
        // Apply HardSwish: x * clamp((x + 3) / 6, 0, 1)
        float hard_swish_factor = fminf(fmaxf((x + 3.0f) / 6.0f, 0.0f), 1.0f);
        output[idx] = x * hard_swish_factor;
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
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_relu_hardswish_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, output, size
    );
}