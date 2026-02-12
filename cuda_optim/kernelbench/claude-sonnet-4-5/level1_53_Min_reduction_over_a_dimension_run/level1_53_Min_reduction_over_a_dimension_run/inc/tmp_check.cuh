__global__ void min_reduction_dim1_kernel_ori(const float* input, float* output, 
                                          int batch_size, int dim1, int dim2) {
    int batch_idx = blockIdx.y;
    int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && dim2_idx < dim2) {
        float min_val = input[batch_idx * dim1 * dim2 + dim2_idx];
        
        for (int i = 1; i < dim1; i++) {
            float val = input[batch_idx * dim1 * dim2 + i * dim2 + dim2_idx];
            min_val = fminf(min_val, val);
        }
        
        output[batch_idx * dim2 + dim2_idx] = min_val;
    }
}

__global__ void min_reduction_dim2_kernel_ori(const float* input, float* output,
                                          int batch_size, int dim1, int dim2) {
    int batch_idx = blockIdx.y;
    int dim1_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && dim1_idx < dim1) {
        float min_val = input[batch_idx * dim1 * dim2 + dim1_idx * dim2];
        
        for (int i = 1; i < dim2; i++) {
            float val = input[batch_idx * dim1 * dim2 + dim1_idx * dim2 + i];
            min_val = fminf(min_val, val);
        }
        
        output[batch_idx * dim1 + dim1_idx] = min_val;
    }
}

__global__ void min_reduction_dim0_kernel_ori(const float* input, float* output,
                                          int batch_size, int dim1, int dim2) {
    int dim1_idx = blockIdx.y;
    int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim1_idx < dim1 && dim2_idx < dim2) {
        float min_val = input[dim1_idx * dim2 + dim2_idx];
        
        for (int i = 1; i < batch_size; i++) {
            float val = input[i * dim1 * dim2 + dim1_idx * dim2 + dim2_idx];
            min_val = fminf(min_val, val);
        }
        
        output[dim1_idx * dim2 + dim2_idx] = min_val;
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
    int dim1 = in_height;
    int dim2 = in_channels;
    int dim = 1;
    
    const int block_size = 256;
    
    if (dim == 0) {
        dim3 grid((dim2 + block_size - 1) / block_size, dim1);
        min_reduction_dim0_kernel_ori<<<grid, block_size, 0, stream>>>(
            input, 
            output,
            batch_size, dim1, dim2
        );
    } else if (dim == 1) {
        dim3 grid((dim2 + block_size - 1) / block_size, batch_size);
        min_reduction_dim1_kernel_ori<<<grid, block_size, 0, stream>>>(
            input, 
            output,
            batch_size, dim1, dim2
        );
    } else if (dim == 2) {
        dim3 grid((dim1 + block_size