__global__ void gelu_kernel_opt(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cube = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * cube); // sqrt(2/pi)
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * val * (1.0f + tanh_val);
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
    int size = in_elems;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel_opt<<<num_blocks, block_size>>>(input, output, size);
}