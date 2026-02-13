__global__ void argmin_kernel_opt(const float* input, int64_t* output, 
                               int batch_size, int dim1, int dim2, int argmin_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (argmin_dim == 1) {
        // argmin along dim1 (middle dimension)
        int total_elements = batch_size * dim2;
        if (idx < total_elements) {
            int b = idx / dim2;
            int k = idx % dim2;
            
            float min_val = input[b * dim1 * dim2 + 0 * dim2 + k];
            int64_t min_idx = 0;
            
            for (int j = 1; j < dim1; j++) {
                float val = input[b * dim1 * dim2 + j * dim2 + k];
                if (val < min_val) {
                    min_val = val;
                    min_idx = j;
                }
            }
            
            output[b * dim2 + k] = min_idx;
        }
    } else if (argmin_dim == 0) {
        // argmin along dim0 (batch dimension)
        int total_elements = dim1 * dim2;
        if (idx < total_elements) {
            int j = idx / dim2;
            int k = idx % dim2;
            
            float min_val = input[0 * dim1 * dim2 + j * dim2 + k];
            int64_t min_idx = 0;
            
            for (int b = 1; b < batch_size; b++) {
                float val = input[b * dim1 * dim2 + j * dim2 + k];
                if (val < min_val) {
                    min_val = val;
                    min_idx = b;
                }
            }
            
            output[j * dim2 + k] = min_idx;
        }
    } else if (argmin_dim == 2) {
        // argmin along dim2 (last dimension)
        int total_elements = batch_size * dim1;
        if (idx < total_elements) {
            int b = idx / dim1;
            int j = idx % dim1;
            
            float min_val = input[b * dim1 * dim2 + j * dim2 + 0];
            int64_t min_idx = 0;
            
            for (int k = 1; k < dim2; k++) {
                float val = input[b * dim1 * dim2 + j * dim2 + k];
                if (val < min_val) {
                    min_val = val;
                    min_idx = k;
                }
            }
            
            output[b * dim1 + j] = min_idx;
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
    // Map tensor dimensions: assume layout is [batch, dim1, dim2]
    int batch_size = in_batch;
    int dim1 = in_height;
    int dim2 = in_width;

    // Determine argmin_dim based on output shape change
    int argmin_dim = -1;
    if (out_batch == dim1 && out_height == dim2) {
        argmin_dim = 0; // reduced batch dim
    } else if (out_batch == batch_size && out_height == dim2) {
        argmin_dim = 1; // reduced height (dim1)
    } else if (out_batch == batch_size && out_height == dim1) {
        argmin_dim = 2; // reduced width (dim2)
    }

    // Determine total threads based on argmin_dim
    int total_threads = 0;
    if (argmin_dim == 0) {
        total_threads = dim1 * dim2;
    } else if (argmin_dim == 1) {
        total_threads = batch_size * dim2;
    } else if (argmin_dim == 2) {
        total_threads = batch_size * dim1;
    }

    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    argmin_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<int64_t*>(output),
        batch_size, dim1, dim2, argmin_dim
    );
}