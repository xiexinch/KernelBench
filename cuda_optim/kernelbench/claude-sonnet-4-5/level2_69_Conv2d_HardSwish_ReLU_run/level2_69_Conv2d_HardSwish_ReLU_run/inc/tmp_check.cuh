__global__ void fused_hardswish_relu_kernel_ori(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // HardSwish: x * relu6(x + 3) / 6
        float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        float hardswish_val = x * relu6_val / 6.0f;
        // ReLU
        output[idx] = fmaxf(hardswish_val, 0.0f);
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
    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;
    fused_hardswish_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems
    );
}