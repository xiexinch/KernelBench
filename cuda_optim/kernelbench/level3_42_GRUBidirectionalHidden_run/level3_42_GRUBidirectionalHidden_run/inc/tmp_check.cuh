__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void gru_cell_forward_kernel_ori(
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int batch_size = in_batch;
    int hidden_size = out_channels;
    const int block_size = BLOCK_SIZE;
    int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    
    // Assuming input layout: [x_input, h_prev, gates_x, gates_h]
    // This is a simplified entry point - actual tensor splitting would be needed
    const float* x_input = input;
    const float* h_prev = input + batch_size * in_channels;
    const float* gates_x = input + 2 * batch_size * in_channels;
    const float* gates_h = input + 2 * batch_size * in_channels + batch_size * hidden_size * 3;
    
    gru_cell_forward_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        x_input,
        h_prev,
        gates_x,
        gates_h,
        output,
        batch_size,
        hidden_size
    );
}