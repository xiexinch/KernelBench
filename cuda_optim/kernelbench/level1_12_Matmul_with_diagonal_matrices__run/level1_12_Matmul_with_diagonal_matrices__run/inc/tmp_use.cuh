__global__ void diag_matmul_kernel_opt(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
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
    int N = in_height;
    int M = in_width;
    
    dim3 block_size(16, 16);
    dim3 num_blocks((M + block_size.x - 1) / block_size.x, 
                    (N + block_size.y - 1) / block_size.y);
    
    diag_matmul_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, 
        input + N, 
        output, 
        N, 
        M
    );
}