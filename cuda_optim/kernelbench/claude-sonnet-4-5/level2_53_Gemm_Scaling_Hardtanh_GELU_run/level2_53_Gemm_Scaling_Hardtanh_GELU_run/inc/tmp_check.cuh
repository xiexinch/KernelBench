__global__ void fused_scale_hardtanh_gelu_kernel_ori(
    const float* input, 
    float* output, 
    int size, 
    float scale, 
    float min_val, 
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Scaling
        float x = input[idx] * scale;
        
        // Hardtanh
        x = fmaxf(min_val, fminf(max_val, x));
        
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float gelu_out = 0.5f * x * (1.0f + tanh_inner);
        
        output[idx] = gelu_out;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    int size = in_elems;
    float scale = 0.5f;
    float min_val = -2.0f;
    float max_val = 2.0f;

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_scale_hardtanh_gelu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, output, size, scale, min_val, max_val
    );
}