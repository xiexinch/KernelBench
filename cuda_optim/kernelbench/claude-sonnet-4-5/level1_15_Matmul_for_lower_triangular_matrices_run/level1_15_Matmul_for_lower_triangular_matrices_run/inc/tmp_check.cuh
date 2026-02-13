__global__ void lower_triangular_matmul_kernel_ori(
    const float* A, 
    const float* B, 
    float* C, 
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only compute lower triangular elements
    if (row < N && col < N && row >= col) {
        float sum = 0.0f;
        
        // For lower triangular matrices:
        // A[row, k] is non-zero only when k <= row
        // B[k, col] is non-zero only when k >= col
        // So we sum from k = col to k = row
        for (int k = col; k <= row; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    } else if (row < N && col < N && row < col) {
        // Upper triangular part should be zero
        C[row * N + col] = 0.0f;
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
    // Assuming square matrices of size N x N
    int N = in_height; // or in_width, since it's square

    dim3 block_size(16, 16);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x, 
                    (N + block_size.y - 1) / block_size.y);
    
    lower_triangular_matmul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output), // Note: In original, B is second input; but interface only gives one input ptr.
        reinterpret_cast<float*>(output),
        N
    );
}