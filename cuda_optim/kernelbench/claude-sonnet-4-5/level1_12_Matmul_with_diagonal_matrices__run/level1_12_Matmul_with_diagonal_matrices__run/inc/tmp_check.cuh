__global__ void diag_matmul_kernel_ori(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
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
    // Input interpretation:
    // - A is 1D tensor of size N -> passed via 'input'
    // - B is 2D tensor of size (N, M) -> also passed via 'input' after A
    // However, kernelbench expects a single input and output buffer.
    // Based on the original kernel logic, we assume:
    //   input[0:in_elems/2] = A (size N)
    //   input[in_elems/2:] = B (size N*M)
    // But the original kernel takes two separate pointers.
    //
    // Since the benchmarking framework provides only one input and one output buffer,
    // and the original kernel requires two inputs (A and B), we reinterpret:
    //   Let A = input
    //   Let B = input + N   (since A has N elements)
    // This matches the original usage where A is size N and B is size N*M,
    // so total input size = N + N*M = N*(M+1)
    //
    // From the problem context: 
    //   A shape: (N,)
    //   B shape: (N, M)
    // So we deduce:
    int N = in_height;  // or in_batch? But given A is 1D (N,), and B is (N,M),
                        // typical call would have in_batch=1, in_height=N, in_width=M, in_channels=1?
    // However, the example call uses:
    //   A = torch.rand(N)      -> size N
    //   B = torch.rand(N, M)   -> size N*M
    // So total input elements = N + N*M
    // But kernelbench passes all inputs in one buffer.
    //
    // Given the function signature constraints and to match original logic,
    // we assume:
    //   N = out_height (since output is N x M)
    //   M = out_width
    int N_val = out_height;
    int M_val = out_width;

    const T* A = input;
    const T* B = input + N_val;  // A has N elements

    dim3 block_size(16, 16);
    dim3 num_blocks((M_val + block_size.x - 1) / block_size.x, 
                    (N_val + block_size.y - 1) / block_size.y);

    diag_matmul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(A),
        reinterpret_cast<const float*>(B),
        reinterpret_cast<float*>(output),
        N_val,
        M_val
    );
}