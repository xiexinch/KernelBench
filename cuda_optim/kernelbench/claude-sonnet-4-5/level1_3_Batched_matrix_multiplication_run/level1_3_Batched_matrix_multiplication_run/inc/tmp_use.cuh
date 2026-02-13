#define TILE_SIZE 16

__global__ void batched_matmul_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int m, int k, int n) {
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int A_offset = batch_idx * m * k;
    int B_offset = batch_idx * k * n;
    int C_offset = batch_idx * m * n;
    
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        int tile_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < m && tile_col < k) {
            tile_A[threadIdx.y][threadIdx.x] = A[A_offset + row * k + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tile_row < k && col < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[B_offset + tile_row * n + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[C_offset + row * n + col] = sum;
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
    // Map input layout to A: [batch, m, k]
    // Map output layout to C: [batch, m, n]
    // B is assumed to be provided as part of the kernel logic, but since we only have input and output,
    // we reinterpret input as containing both A and B concatenated.
    // However, based on the original kernel signature, we need separate A and B.
    // Since the test function only provides one input pointer, we assume that:
    // - The first half of input corresponds to A (size: in_batch * in_height * in_channels)
    // - The second half corresponds to B (size: in_batch * in_channels * in_width)
    // But the given parameters don't expose B's dimensions directly.
    //
    // Instead, we follow the original kernel's intended usage:
    // A: [in_batch, in_height, in_channels] -> m=in_height, k=in_channels
    // B: [in_batch, in_channels, in_width]  -> k=in_channels, n=in_width
    // C: [out_batch, out_height, out_width] -> m=out_height, n=out_width
    //
    // We assume in_batch == out_batch, in_height == out_height, in_width == out_width

    int batch_size = in_batch;
    int m = in_height;
    int k = in_channels;
    int n = in_width;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, 
              (m + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    // Assume input contains A and B concatenated:
    // A starts at input
    // B starts at input + (batch_size * m * k)
    T* A = input;
    T* B = input + (batch_size * m * k);

    batched_matmul_kernel_opt<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(A),
        reinterpret_cast<const float*>(B),
        reinterpret_cast<float*>(output),
        batch_size, m, k, n
    );
}