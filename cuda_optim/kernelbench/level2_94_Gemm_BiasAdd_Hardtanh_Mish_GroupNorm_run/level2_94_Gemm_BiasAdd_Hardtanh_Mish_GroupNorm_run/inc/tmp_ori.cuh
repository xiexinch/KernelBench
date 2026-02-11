__device__ float hardtanh_activation(float x) {
    return fminf(fmaxf(x, -1.0f), 1.0f);
}

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_bias_hardtanh_mish_kernel_opt(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * features;
    
    if (idx < total_elements) {
        int feature_idx = idx % features;
        float val = input[idx] + bias[feature_idx];
        val = hardtanh_activation(val);
        val = mish_activation(val);
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
    int features = in_channels;
    
    const int threads = 256;
    const int blocks = (batch_size * features + threads - 1) / threads;
    
    fused_bias_hardtanh_mish_kernel_opt<<<blocks, threads, 0, stream>>>(
        input,
        input + in_elems,
        output,
        batch_size,
        features
    );
}