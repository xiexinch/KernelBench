__global__ void fused_div_leaky_relu_kernel_ori(const float* input, float* output, 
                                             int size, float divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        output[idx] = val > 0.0f ? val : val * negative_slope;
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
    float divisor = 2.0f;
    float negative_slope = 0.01f;
    int size = in_elems;

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leaky_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        size,
        divisor,
        negative_slope
    );
}