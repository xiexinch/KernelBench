#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void fused_max_mean_sub_gelu_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int features
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_data = input + row * features;
    
    // Find max using reduction
    __shared__ float shared_max[BLOCK_SIZE];
    float thread_max = -FLT_MAX;
    
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_data[i]);
    }
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    float max_val = shared_max[0];
    
    // Compute mean of max_val (which is just max_val since it's a single value per row)
    float mean_val = max_val;
    
    // Subtract and apply GELU
    float result = max_val - mean_val; // This will be 0
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = result * result * result;
    float tanh_arg = sqrt_2_over_pi * (result + coeff * x3);
    float tanh_val = tanhf(tanh_arg);
    float gelu_result = 0.5f * result * (1.0f + tanh_val);
    
    if (threadIdx.x == 0) {
        output[row] = gelu_result;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Map 4D tensor layout to 2D kernel parameters
    // Assumes input is (batch, 1, 1, features) and output is (batch, 1, 1, 1)
    int batch_size = in_batch;
    int features = in_width;
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    fused_max_mean_sub_gelu_kernel_opt<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        features
    );
}