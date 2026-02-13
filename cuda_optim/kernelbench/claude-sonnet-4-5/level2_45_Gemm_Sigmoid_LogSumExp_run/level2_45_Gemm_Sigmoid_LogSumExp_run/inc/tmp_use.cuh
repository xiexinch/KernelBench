#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void matmul_sigmoid_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int input_size, int hidden_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        sum += bias[col];
        output[row * hidden_size + col] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void matmul_logsumexp_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int hidden_size, int output_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        float max_val = -INFINITY;
        
        // First pass: find max for numerical stability
        for (int col = 0; col < output_size; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                sum += input[row * hidden_size + i] * weight[col * hidden_size + i];
            }
            sum += bias[col];
            max_val = fmaxf(max_val, sum);
        }
        
        // Second pass: compute logsumexp
        float exp_sum = 0.0f;
        for (int col = 0; col < output_size; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                sum += input[row * hidden_size + i] * weight[col * hidden_size + i];
            }
            sum += bias[col];
            exp_sum += expf(sum - max_val);
        }
        
        output[row] = max_val + logf(exp_sum);
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
    // For this benchmark, we assume the following dimension mapping:
    // Input tensor: [in_batch, in_width] where in_height = in_channels = 1
    // Output tensor shape determines which kernel to run
    
    // Case 1: Output is 1D (logsumexp) - out_height, out_channels, out_width are 1
    if (out_height == 1 && out_channels == 1 && out_width == 1) {
        int batch_size = in_batch;
        int hidden_size = in_width; // input is [batch, hidden]
        int output_size = out_batch; // but we need to infer from context
        
        // Since weight/bias dimensions aren't provided in the interface,
        // we use the problem's example configuration:
        // From the original code: input_size=2048, hidden_size=4096, output_size=1024
        // But for general case, we assume output_size can be derived from out_batch
        // However, out_batch should equal batch_size for logsumexp
        // So we need another way - we'll use a fixed output_size based on typical usage
        // But this is not robust. Instead, we note that in the original test:
        // - First kernel: input [16384, 2048] -> output [16384, 4096]
        // - Second kernel: input [16384, 4096] -> output [16384]
        // So for logsumexp, hidden_size = in_width, and output_size is unknown
        
        // Given the constraints of the interface, we must assume that
        // additional parameters (weight, bias) are available as global device variables
        extern __device__ float* d_weight;
        extern __device__ float* d_bias;
        extern __device__ int d_output_size;
        
        int block_size = 256;
        int num_blocks = (batch_size + block_size - 1) / block_size;
        
        matmul_logsumexp_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            input, d_weight, d_bias, output,
            batch_size, hidden_size, d_output_size
        );
    }
    // Case 2: Output is 2D (sigmoid) - typical matmul case
    else {
        int batch_size = in_batch;
        int input_size = in_width;
        int hidden_size = out_width; // output is [batch, hidden]
        
        extern __device__ float* d_weight;
        extern __device__ float* d_bias;
        
        dim3 block(16, 16);
        dim3 grid((hidden_size + block.x - 1) / block.x, 
                  (batch_size + block.y - 1) / block.y);
        
        matmul_sigmoid_kernel_opt<<<grid, block, 0, stream>>>(
            input, d_weight, d_bias, output,
            batch_size, input_size, hidden_size
        );
    }
}