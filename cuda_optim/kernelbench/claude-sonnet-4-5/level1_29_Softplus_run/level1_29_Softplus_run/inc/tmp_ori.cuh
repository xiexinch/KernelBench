__global__ void softplus_kernel_opt(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // For numerical stability: if x > 20, softplus(x) ≈ x
        // Otherwise, softplus(x) = log(1 + exp(x))
        if (val > 20.0f) {
            out[idx] = val;
        } else {
            out[idx] = log1pf(expf(val));
        }
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
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    softplus_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems
    );
}