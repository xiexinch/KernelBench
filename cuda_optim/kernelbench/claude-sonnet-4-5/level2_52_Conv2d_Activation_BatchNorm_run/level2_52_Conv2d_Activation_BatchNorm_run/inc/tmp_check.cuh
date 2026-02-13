__global__ void fused_activation_kernel_ori(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // softplus(x) = log(1 + exp(x))
        // For numerical stability
        float softplus_x;
        if (x > 20.0f) {
            softplus_x = x;
        } else if (x < -20.0f) {
            softplus_x = expf(x);
        } else {
            softplus_x = logf(1.0f + expf(x));
        }
        // tanh(softplus(x))
        float tanh_softplus = tanhf(softplus_x);
        // x * tanh(softplus(x))
        output[idx] = x * tanh_softplus;
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
    fused_activation_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems
    );
}