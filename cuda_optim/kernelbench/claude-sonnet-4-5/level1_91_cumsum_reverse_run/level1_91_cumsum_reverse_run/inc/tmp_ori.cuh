__global__ void reverse_cumsum_kernel_dim1_opt(const float* input, float* output, 
                                            int batch_size, int feature_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int offset = batch_idx * feature_size;
        float sum = 0.0f;
        
        // Iterate from the end to the beginning
        for (int i = feature_size - 1; i >= 0; i--) {
            sum += input[offset + i];
            output[offset + i] = sum;
        }
    }
}

__global__ void reverse_cumsum_kernel_dim0_opt(const float* input, float* output, 
                                            int batch_size, int feature_size) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feat_idx < feature_size) {
        float sum = 0.0f;
        
        // Iterate from the end to the beginning
        for (int i = batch_size - 1; i >= 0; i--) {
            sum += input[i * feature_size + feat_idx];
            output[i * feature_size + feat_idx] = sum;
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
    int feature_size = in_height;
    int dim = 1;
    
    const int block_size = 256;
    
    if (dim == 1) {
        const int num_blocks = (batch_size + block_size - 1) / block_size;
        reverse_cumsum_kernel_dim1_opt<<<num_blocks, block_size, 0, stream>>>(
            input, 
            output, 
            batch_size, 
            feature_size
        );
    } else if (dim == 0) {
        const int num_blocks = (feature_size + block_size - 1) / block_size;
        reverse_cumsum_kernel_dim0_opt<<<num_blocks, block_size, 0, stream>>>(
            input, 
            output, 
            batch_size, 
            feature_size
        );
    }
}