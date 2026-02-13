#define TILE_SIZE 32

__global__ void matmul_kernel_ori(const float* A, const float* B, float* C, 
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
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            shared_A[ty][tx] = A[row * K + a_col];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            shared_B[ty][tx] = B[b_row * N + col];
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Interpret input as matrix A (M x K) and assume B is stored right after A in input
    // From the original torch code: A is (M, K), B is (K, N)
    // So total input size = M*K + K*N
    // But this function only receives one input pointer.
    // Based on kernelbench convention and example, we reinterpret:
    //   input[0:in_elems/2] -> A
    //   input[in_elems/2:end] -> B
    // However, original matmul takes two separate tensors.
    // To match the given signature, we assume:
    //   A is of shape (M, K) = (in_batch, in_height)  --> but this doesn't align
    //
    // Instead, follow the logic from the provided Torch code:
    //   A: (M, K)
    //   B: (K, N)
    //   C: (M, N)
    //
    // We map:
    //   M = in_batch
    //   K = in_height
    //   N = out_width
    //
    // And assume input contains A followed by B.
    // Therefore:
    //   A = input
    //   B = input + (M * K)
    //   C = output

    int M = in_batch;
    int K = in_height;
    int N = out_width;

    const float* A = reinterpret_cast<const float*>(input);
    const float* B = reinterpret_cast<const float*>(input + M * K);
    float* C = reinterpret_cast<float*>(output);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel_ori<<<grid_size, block_size, 0, stream>>>(
        A, B, C, M, K, N
    );
}