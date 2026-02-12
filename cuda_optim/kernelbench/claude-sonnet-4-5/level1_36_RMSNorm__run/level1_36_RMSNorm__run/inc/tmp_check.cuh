__global__ void rmsnorm_kernel_ori(const float* x, float* out, int batch_size, int features, 
                                int dim1, int dim2, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * dim1 * dim2;
    
    if (idx < total_elements) {
        int b = idx / (dim1 * dim2);
        int remaining = idx % (dim1 * dim2);
        int d1 = remaining / dim2;
        int d2 = remaining % dim2;
        
        // Calculate RMS for this position across features
        float sum_sq = 0.0f;
        int base_offset = b * features * dim1 * dim2 + d1 * dim2 + d2;
        
        for (int f = 0; f < features; f++) {
            int offset = base_offset + f * dim1 * dim2;
            float val = x[offset];
            sum_sq += val * val;
        }
        
        float rms = sqrtf(sum_sq / features + eps);
        
        // Normalize all features at this position
        for (int f = 0; f < features; f++) {
            int offset = base_offset + f * dim1 * dim2;
            out[offset] = x[offset] / rms;
        }
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
    int features = in_channels;
    int dim1 = in_height;
    int dim2 = in_width;
    float eps = 1e-5;
    
    int total_elements = batch_size * dim1 * dim2;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    rmsnorm_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, 
        output, 
        batch_size, 
        features, 
        dim1, 
        dim2, 
        eps
    );
}