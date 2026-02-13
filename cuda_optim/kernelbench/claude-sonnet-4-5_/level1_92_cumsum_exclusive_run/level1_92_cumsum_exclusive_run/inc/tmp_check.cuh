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
    int64_t outer_size = in_batch;
    int64_t dim_size = in_height;
    int64_t inner_size = in_channels * in_width;
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    exclusive_cumsum_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        output,
        outer_size,
        dim_size,
        inner_size
    );
}