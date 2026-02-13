#include <climits>
#include <cfloat>
#include <cuda_runtime.h>

__global__ void argmax_dim1_kernel_opt(const float* input, int64_t* output, 
                                     int batch_size, int dim1, int dim2) {
    int batch_idx = blockIdx.y;
    int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && dim2_idx < dim2) {
        float max_val = -INFINITY;
        int64_t max_idx = 0;
        
        for (int i = 0; i < dim1; i++) {
            int input_idx = batch_idx * dim1 * dim2 + i * dim2 + dim2_idx;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        int output_idx = batch_idx * dim2 + dim2_idx;
        output[output_idx] = max_idx;
    }
}

__global__ void argmax_dim2_kernel_opt(const float* input, int64_t* output, 
                                     int batch_size, int dim1, int dim2) {
    int batch_idx = blockIdx.y;
    int dim1_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && dim1_idx < dim1) {
        float max_val = -INFINITY;
        int64_t max_idx = 0;
        
        for (int i = 0; i < dim2; i++) {
            int input_idx = batch_idx * dim1 * dim2 + dim1_idx * dim2 + i;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        int output_idx = batch_idx * dim1 + dim1_idx;
        output[output_idx] = max_idx;
    }
}

__global__ void argmax_dim0_kernel_opt(const float* input, int64_t* output, 
                                     int batch_size, int dim1, int dim2) {
    int dim1_idx = blockIdx.y;
    int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim1_idx < dim1 && dim2_idx < dim2) {
        float max_val = -INFINITY;
        int64_t max_idx = 0;
        
        for (int i = 0; i < batch_size; i++) {
            int input_idx = i * dim1 * dim2 + dim1_idx * dim2 + dim2_idx;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        int output_idx = dim1_idx * dim2 + dim2_idx;
        output[output_idx] = max_idx;
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
    // Map input tensor shape to 3D: [batch_size, dim1, dim2]
    // Assume the input is 3D and dim corresponds to one of the dimensions
    // Based on the original code, we assume:
    //   in_batch = batch_size
    //   in_height = dim1
    //   in_width = dim2
    //   in_channels is unused (original code uses 3D tensors without channels)

    int batch_size = in_batch;
    int dim1 = in_height;
    int dim2 = in_width;

    const int block_size = 256;
    int dim = -1;

    // Infer reduction dimension based on output shape
    if (out_batch == dim1 && out_height == dim2 && out_channels == 1 && out_width == 1) {
        // Output shape [dim1, dim2] => reduction over dim0 (batch)
        dim = 0;
    } else if (out_batch == batch_size && out_height == dim2 && out_channels == 1 && out_width == 1) {
        // Output shape [batch_size, dim2] => reduction over dim1
        dim = 1;
    } else if (out_batch == batch_size && out_height == dim1 && out_channels == 1 && out_width == 1) {
        // Output shape [batch_size, dim1] => reduction over dim2
        dim = 2;
    }

    if (dim == 0) {
        dim3 num_blocks((dim2 + block_size - 1) / block_size, dim1);
        argmax_dim0_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<int64_t*>(output),
            batch_size, dim1, dim2);
    } else if (dim == 1) {
        dim3 num_blocks((dim2 + block_size - 1) / block_size, batch_size);
        argmax_dim1_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<int64_t*>(output),
            batch_size, dim1, dim2);
    } else if (dim == 2) {
        dim3 num_blocks((dim1 + block_size - 1) / block_size, batch_size);
        argmax_dim2_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<int64_t*>(output),
            batch_size, dim1, dim2);
    }
}