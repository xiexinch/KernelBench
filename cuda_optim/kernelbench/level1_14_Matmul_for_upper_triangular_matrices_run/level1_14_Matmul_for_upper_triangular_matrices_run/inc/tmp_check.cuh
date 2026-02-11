__global__ void upper_triangular_matmul_kernel_ori(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only compute upper triangular elements
    if (row < N && col < N && col >= row) {
        float sum = 0.0f;
        // For upper triangular matrices A and B:
        // C[i][j] = sum(A[i][k] * B[k][j]) for k in range
        // A[i][k] is non-zero only when k >= i
        // B[k][j] is non-zero only when j >= k
        // So k ranges from i to j (inclusive)
        for (int k = row; k <= col; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else if (row < N && col < N && col < row) {
        // Set lower triangular elements to zero
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
    const int N = in_height;
    const float* A = input;
    const float* B = input + N * N;
    float* C = output;
    
    const dim3 block_size(16, 16);
    const dim3 num_blocks((N + block_size.x - 1) / block_size.x,
                          (N + block_size.y - 1) / block_size.y);
    
    upper_triangular_matmul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        A,
        B,
        C,
        N
    );
}