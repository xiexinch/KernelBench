__device__ float gelu_kernel(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_add_min_gelu_mul_kernel_opt(
    const float* input,
    float* output,
    const float add_value,
    const float multiply_value,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Add
        val = val + add_value;
        // Min with 0
        val = fminf(val, 0.0f);
        // GELU
        val = gelu_kernel(val);
        // Multiply
        val = val * multiply_value;
        output[idx] = val;
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
    float add_value = 0.5;
    float multiply_value = 2.0;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_add_min_gelu_mul_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        add_value,
        multiply_value,
        size
    );
}