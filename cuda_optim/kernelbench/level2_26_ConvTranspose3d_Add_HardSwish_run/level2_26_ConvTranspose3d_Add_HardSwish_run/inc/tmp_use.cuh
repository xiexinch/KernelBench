__device__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_add_hardswish_kernel_opt(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = conv_out[idx] + add_input[idx];
        output[idx] = val * hardswish(val);
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
    
    fused_add_hardswish_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        input + size,
        output,
        size
    );
}