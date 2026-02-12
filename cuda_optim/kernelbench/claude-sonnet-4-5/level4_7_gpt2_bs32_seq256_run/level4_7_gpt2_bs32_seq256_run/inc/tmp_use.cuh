__global__ void gelu_kernel_opt(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void layernorm_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_size,
    float eps
) {
    int idx = blockIdx.x;
    if (idx < batch_size) {
        const float* in_ptr = input + idx * hidden_size;
        float* out_ptr = output + idx * hidden_size;
        
        // Calculate mean
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += in_ptr[i];
        }
        float mean = sum / hidden_size;
        
        // Calculate variance
        float var_sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = in_ptr[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden_size;
        float inv_std = 1.0f / sqrtf(var + eps);
        
        // Normalize and apply affine transformation
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
    }
}

__global__ void fused_residual_gelu_kernel_opt(
    const float* input,
    const float* residual,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] + residual[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
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
    int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(input, output, size);
}