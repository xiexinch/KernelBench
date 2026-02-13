__global__ void selu_kernel_ori(const float* x, float* out, int size) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = scale * (val > 0.0f ? val : alpha * (expf(val) - 1.0f));
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
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    selu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(input, output, in_elems);
}