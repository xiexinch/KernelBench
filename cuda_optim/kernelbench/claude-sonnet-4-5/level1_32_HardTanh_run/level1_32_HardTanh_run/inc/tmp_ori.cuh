__global__ void hardtanh_kernel_opt(const float* x, float* out, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = fminf(fmaxf(val, min_val), max_val);
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
    float min_val = -1.0;
    float max_val = 1.0;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardtanh_kernel_opt<<<num_blocks, block_size>>>(input, output, size, min_val, max_val);
}