__global__ void elu_kernel_ori(const float* x, float* out, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0.0f ? val : alpha * (expf(val) - 1.0f);
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
    float alpha = 1.0f;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    elu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(input, output, size, alpha);
}