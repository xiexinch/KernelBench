__global__ void exclusive_cumsum_kernel_ori(const float* input, float* output, 
                                       int outer_size, int dim_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outer_size * inner_size) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        
        float sum = 0.0f;
        for (int i = 0; i < dim_size; i++) {
            int input_idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            int output_idx = input_idx;
            output[output_idx] = sum;
            sum += input[input_idx];
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
    // For this kernel, we assume the cumsum is applied along the last dimension (dim = -1 or equivalent)
    // Based on the original logic, we need to compute outer_size, inner_size, and dim_size
    // Since the original code uses a generic dim, but our interface doesn't pass it,
    // we infer from the shape: assume cumsum is along the innermost dimension (width)

    int outer_size = in_batch * in_height * in_channels;
    int inner_size = 1;
    int dim_size = in_width;

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    exclusive_cumsum_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        outer_size,
        dim_size,
        inner_size
    );
}