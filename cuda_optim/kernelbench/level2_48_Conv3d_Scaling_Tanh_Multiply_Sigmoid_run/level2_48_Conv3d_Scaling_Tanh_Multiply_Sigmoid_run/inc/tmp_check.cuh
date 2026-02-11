__global__ void fused_ops_kernel_ori(
    const float* input,
    const float* scaling_factor,
    const float* bias,
    float* output,
    int size,
    int out_channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int channel = (idx / spatial_size) % out_channels;
        float val = input[idx];
        val = val * scaling_factor[channel];
        val = tanhf(val);
        val = val * bias[channel];
        val = 1.0f / (1.0f + expf(-val));
        output[idx] = val;
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
    int size = out_elems;
    int spatial_size = out_height * out_width;
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_ops_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        input + size,
        input + size + out_channels,
        output,
        size,
        out_channels,
        spatial_size
    );
}