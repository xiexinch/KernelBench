#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256




__global__ void gru_cell_forward_kernel_opt(
    const float* __restrict__ x_input,
    const float* __restrict__ h_prev,
    const float* __restrict__ gates_x,
    const float* __restrict__ gates_h,
    float* __restrict__ h_out,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * hidden_size;
    
    if (idx < total_size) {
        int b = idx / hidden_size;
        int h = idx % hidden_size;
        
        int offset = b * hidden_size * 3 + h;
        
        // Reset gate
        float r = sigmoid(gates_x[offset] + gates_h[offset]);
        
        // Update gate
        float z = sigmoid(gates_x[offset + hidden_size] + gates_h[offset + hidden_size]);
        
        // New gate
        float n = tanh_activation(gates_x[offset + 2 * hidden_size] + 
                                   r * gates_h[offset + 2 * hidden_size]);
        
        // New hidden state
        float h_prev_val = h_prev[idx];
        h_out[idx] = (1.0f - z) * n + z * h_prev_val;
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
    int batch_size = in_batch;
    int hidden_size = in_width;
    int total_size = batch_size * hidden_size;

    const float* gates_x = reinterpret_cast<const float*>(input);
    const float* h_prev = reinterpret_cast<const float*>(output);
    const float* gates_h = reinterpret_cast<const float*>(input) + batch_size * hidden_size * 3;
    float* h_out = reinterpret_cast<float*>(output);

    const int block_size = BLOCK_SIZE;
    int num_blocks = (total_size + block_size - 1) / block_size;

    gru_cell_forward_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        nullptr,
        h_prev,
        gates_x,
        gates_h,
        h_out,
        batch_size,
        hidden_size
    );
}