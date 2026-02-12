__global__ void matmul_kernel_ori(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
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
    int M = in_batch;
    int N = in_height;
    int K = out_height;
    
    dim3 block_size(16, 16);
    dim3 num_blocks((K + block_size.x - 1) / block_size.x,
                    (M + block_size.y - 1) / block_size.y);
    
    matmul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        input + in_elems,
        output,
        M, N, K
    );
}