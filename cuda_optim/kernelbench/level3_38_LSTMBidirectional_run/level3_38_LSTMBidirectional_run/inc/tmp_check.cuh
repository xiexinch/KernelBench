__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_func(float x) {
    return tanhf(x);
}

__global__ void lstm_cell_forward_kernel_ori(
    const float* x,
    const float* h_prev,
    const float* c_prev,
    const float* weight_ih,
    const float* weight_hh,
    const float* bias_ih,
    const float* bias_hh,
    float* h_new,
    float* c_new,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x;
    int h = threadIdx.x;
    
    if (b < batch_size && h < hidden_size) {
        // Compute gates: i, f, g, o
        float gates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Input contribution
        for (int gate = 0; gate < 4; gate++) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += x[b * input_size + i] * weight_ih[(gate * hidden_size + h) * input_size + i];
            }
            gates[gate] = sum + bias_ih[gate * hidden_size + h];
        }
        
        // Hidden contribution
        for (int gate = 0; gate < 4; gate++) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_size; i++) {
                sum += h_prev[b * hidden_size + i] * weight_hh[(gate * hidden_size + h) * hidden_size + i];
            }
            gates[gate] += sum + bias_hh[gate * hidden_size + h];
        }
        
        float i_gate = sigmoid(gates[0]);
        float f_gate = sigmoid(gates[1]);
        float g_gate = tanh_func(gates[2]);
        float o_gate = sigmoid(gates[3]);
        
        float c_val = f_gate * c_prev[b * hidden_size + h] + i_gate * g_gate;
        float h_val = o_gate * tanh_func(c_val);
        
        c_new[b * hidden_size + h] = c_val;
        h_new[b * hidden_size + h] = h_val;
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
    int input_size = in_channels;
    int hidden_size = out_channels;
    
    const float* x = input;
    const float* h_prev = input + batch_size * input_size;
    const float* c_prev = input + batch_size * input_size + batch_size * hidden_size;
    const float* weight_ih = input + batch_size * input_size + 2 * batch_size * hidden_size;
    const float* weight_hh = weight_ih + 4 * hidden_size * input_size;
    const float* bias_ih = weight_hh + 4 * hidden_size * hidden_size;
    const float* bias_hh = bias_ih + 4 * hidden_size;
    
    float* h_new = output;
    float* c_new = output + batch_size * hidden_size;
    
    dim3 grid(batch_size);
    dim3 block(hidden_size);
    
    lstm_cell_forward_kernel_ori<<<grid, block, 0, stream>>>(
        x, h_