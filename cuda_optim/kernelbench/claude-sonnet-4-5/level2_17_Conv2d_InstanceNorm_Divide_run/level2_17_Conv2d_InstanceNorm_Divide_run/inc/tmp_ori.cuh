__global__ void fused_instance_norm_div_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float divide_by,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels;
    
    if (idx < total) {
        int b = idx / channels;
        int c = idx % channels;
        
        const float* input_ptr = input + (b * channels + c) * spatial_size;
        float* output_ptr = output + (b * channels + c) * spatial_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input_ptr[i];
        }
        float mean = sum / spatial_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            float diff = input_ptr[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / spatial_size;
        float inv_std = rsqrtf(variance + eps);
        
        // Normalize, apply affine transform, and divide
        float g = gamma[c];
        float b_val = beta[c];
        for (int i = 0; i < spatial_size; i++) {
            float normalized = (input_ptr[i] - mean) * inv_std;
            output_ptr[i] = (g * normalized + b_val) / divide_by;
        }
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
    int spatial_size = in_height * in_width;
    float divide_by = 2.0f;
    float eps = 1e-5f;

    const int threads = 256;
    int total = batch_size * channels;
    int blocks = (total + threads - 1) / threads;

    fused_instance_norm_div_kernel_opt<<<blocks, threads, 0, stream>>>(
        input,
        static_cast<const float*>(static_cast<const void*>(input)) + in_batch * in_channels * in_height * in_width, // dummy placeholder for gamma
        static_cast<const float*>(static_cast<const void*>(input)) + in_batch * in_channels * in_height * in_width + in_channels, // dummy placeholder for beta
        output,
        batch_size,
        channels,
        spatial_size,
        divide_by,
        eps
    );
}