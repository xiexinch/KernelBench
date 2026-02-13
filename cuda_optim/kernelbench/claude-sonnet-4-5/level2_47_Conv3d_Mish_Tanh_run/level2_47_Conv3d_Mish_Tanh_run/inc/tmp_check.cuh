__global__ void fused_mish_tanh_kernel_ori(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        float softplus = log1pf(expf(x));
        float mish = x * tanhf(softplus);
        // Apply tanh on top of mish
        output[idx] = tanhf(mish);
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
    fused_mish_tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        size
    );
}