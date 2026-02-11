__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_activations_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        // Calculate channel index for bias
        int spatial_size = depth * height * width;
        int temp = idx / spatial_size;
        int channel_idx = temp % channels;
        
        float val = input[idx];
        
        // ReLU
        val = fmaxf(val, 0.0f);
        
        // LeakyReLU with negative_slope=0.01
        val = (val > 0.0f) ? val : (0.01f * val);
        
        // GELU
        val = gelu_activation(val);
        
        // Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        
        // Add bias
        val = val + bias[channel_idx];
        
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
    int batch_size = in_batch;
    int channels = in_channels;
    int depth = in_height;
    int height = in_width;
    int width = out_width;
    int total_size = in_elems;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_activations_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        output,
        batch_size,
        channels,
        depth,
        height,
        width
    );
}