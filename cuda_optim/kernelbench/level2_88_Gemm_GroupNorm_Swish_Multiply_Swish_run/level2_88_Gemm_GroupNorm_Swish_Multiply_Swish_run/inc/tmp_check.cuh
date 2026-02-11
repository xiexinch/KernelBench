__global__ void groupnorm_swish_kernel_ori(
    const float* x, 
    const float* gamma, 
    const float* beta,
    float* out, 
    int batch_size,
    int channels,
    int group_size,
    int num_groups,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < total_groups) {
        int b = idx / num_groups;
        int g = idx % num_groups;
        
        // Calculate mean
        float sum = 0.0f;
        int start_c = g * group_size;
        int end_c = start_c + group_size;
        
        for (int c = start_c; c < end_c; c++) {
            sum += x[b * channels + c];
        }
        float mean = sum / group_size;
        
        // Calculate variance
        float var_sum = 0.0f;
        for (int c = start_c; c < end_c; c++) {
            float diff = x[b * channels + c] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / group_size;
        float std = sqrtf(var + eps);
        
        // Normalize, scale, shift, and apply swish
        for (int c = start_c; c < end_c; c++) {
            float normalized = (x[b * channels + c] - mean) / std;
            float scaled = normalized * gamma[c] + beta[c];
            float sigmoid_val = 1.0f / (1.0f + expf(-scaled));
            out[b * channels + c] = scaled * sigmoid_val;
        }
    }
}

__global__ void multiply_swish_kernel_ori(
    const float* x,
    const float* weight,
    float* out,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = x[idx] * weight[idx % (size / (size / (int)blockDim.x + 1) + 1)];
        int channel = idx % (size / ((size + 255) / 256));
        float weighted = x[idx] * weight[channel];
        float sigmoid_val = 1.0f / (1.0f + expf(-weighted));
        out[idx] = weighted * sigmoid_val;
    }
}

__global__ void multiply_swish_kernel_v2_ori(
    const float* x,
    const float* weight,
    float* out,
    int batch_size,
    int channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels;
    
    if (idx < total_size) {
        int c = idx % channels;
        float weighted = x[idx] * weight[c];
        float sigmoid_val = 1.0f / (1.0f + expf(-weighted));
        out[idx] = weighted * sigmoid_val;
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
    int batch_size = in_batch;
    int channels = in_channels;
    int num_groups = 32;
    int group_size = channels / num_groups;
    float eps = 1e-5;
    
    const int block_size = 256;
    const int num_blocks = (batch_size * num_groups + block_size - 1) / block_size;
    
    T* gamma = nullptr;
    T* beta = nullptr;
    T* temp_output