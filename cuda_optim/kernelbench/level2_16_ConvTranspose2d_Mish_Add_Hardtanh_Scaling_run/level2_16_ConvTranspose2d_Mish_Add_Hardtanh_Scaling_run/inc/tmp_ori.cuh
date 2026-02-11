__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_mish_add_hardtanh_scale_kernel_opt(
    const float* input,
    float* output,
    const float add_value,
    const float scale,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Apply Mish
        val = mish_activation(val);
        // Add value
        val = val + add_value;
        // Apply Hardtanh (clamp between -1 and 1)
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        // Scale
        val = val * scale;
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
    float scale = 2.0;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_mish_add_hardtanh_scale_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        add_value,
        scale,
        size
    );
}