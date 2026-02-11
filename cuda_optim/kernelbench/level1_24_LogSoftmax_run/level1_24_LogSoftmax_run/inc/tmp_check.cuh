#define BLOCK_SIZE 256

__global__ void log_softmax_kernel_ori(const float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* x = input + row * dim;
    float* out = output + row * dim;
    
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Find maximum value in the row
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x[i]);
    }
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_sum += expf(x[i] - max_val);
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = shared_sum[0];
    float log_sum_exp = logf(sum_exp);
    __syncthreads();
    
    // Compute log_softmax
    for (int i = tid; i < dim; i += blockDim.x) {
        out[i] = x[i] - max_val - log_sum_exp;
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
    int batch_size = in_batch;
    int feature_dim = in_elems / in_batch;
    
    const int threads = BLOCK_SIZE;
    const int blocks = batch_size;
    
    log_softmax_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        output,
        batch_size,
        feature_dim
    );
}