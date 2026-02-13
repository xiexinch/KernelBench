__global__ void tanh_kernel_ori(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(x[idx]);
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
    tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(input, output, in_elems);
}