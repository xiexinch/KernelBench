#define TILE_SIZE 32

__global__ void matmul_kernel_opt(const float* A, const float* B, float* C, 
                              int M, int K, int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            shared_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if ((t * TILE_SIZE + ty) < K && col < N) {
            shared_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
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
    // Interpret input layout as matrix A (M x K)
    // and assume a second matrix B is packed right after A in input
    // However, since the original kernel only uses two inputs (A and B),
    // and the provided interface only gives one input pointer,
    // we reinterpret based on typical usage in the example:
    // A is (M, K), B is (K, N), so total input size = M*K + K*N
    // But the function signature doesn't give us B separately.
    //
    // Given the constraints and example, we assume:
    // - input points to A (M x K)
    // - B is not passed; however, original code expects both A and B.
    //
    // Since the problem states "keep original kernel logic" and provides
    // only one input and one output, we must reinterpret parameters.
    //
    // From the example: M = in_height, K = in_width, N = out_width
    // and it's assumed that B is stored contiguously after A,
    // but our interface doesn't support that.
    //
    // However, looking at the provided PyTorch example:
    //   A is (M, K), B is (K, N)
    //   C is (M, N)
    // The test function signature includes:
    //   in_batch, in_height, in_channels, in_width
    //   out_batch, out_height, out_channels, out_width
    //
    // We deduce:
    //   M = in_height (or in_batch * in_height if batched, but original is 2D)
    //   K = in_width
    //   N = out_width
    // And we assume non-batched, non-channel case (as in original kernel).
    //
    // Therefore, we map:
    int M = in_height;
    int K = in_width;
    int N = out_width;

    // Assume input layout: [A (M*K), B (K*N)]
    // So B starts at input + M*K
    const T* A = input;
    const T* B = input + M * K;
    T* C = output;

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                    (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(A),
        reinterpret_cast<const float*>(B),
        reinterpret_cast<float*>(C),
        M, K, N
    );
}