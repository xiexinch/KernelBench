__global__ void batchnorm_relu_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    
    if (idx < total_size) {
        int c = (idx / (H * W)) % C;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float scaled = normalized * w + b;
        output[idx] = fmaxf(0.0f, scaled);
    }
}

__global__ void add_relu_kernel_ori(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(0.0f, a[idx] + b[idx]);
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
    // Determine which kernel to launch based on dimensions
    // For batchnorm_relu: input and output shapes match, and we need extra parameters
    // Since we don't have weight/bias/running stats in signature, assume fused add+relu path
    // as it only requires two inputs and one output with matching sizes.

    // Given the function signature constraints, we map to add_relu when in/out shapes match
    // and treat 'input' as 'a', and reinterpret 'output' as both 'b' and final 'out'
    // However, this is ambiguous. Following the example pattern, we choose one kernel.
    // Based on typical usage in the provided model, we'll implement add_relu path,
    // assuming that 'input' is 'a', and 'output' initially holds 'b', then becomes result.

    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;

    add_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,  // treated as second input 'b'
        output,  // output overwritten in-place
        in_elems
    );
}