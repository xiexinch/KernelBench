__global__ void fused_qk_mask_relu_kernel_ori(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ att,
    const float scale,
    const int B, const int H, const int T, const int hs
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int t_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t_row >= T) return;
    
    const int qk_offset = b * H * T * hs + h * T * hs;
    const int att_offset = b * H * T * T + h * T * T;
    
    for (int t_col = 0; t_col < T; ++t_col) {
        float sum = 0.0f;
        
        // Compute dot product between q[t_row] and k[t_col]
        for (int i = 0; i < hs; ++i) {
            sum += q[qk_offset + t_row * hs + i] * k[qk_offset + t_col * hs + i];
        }
        
        // Scale
        sum *= scale;
        
        // Apply causal mask and ReLU
        if (t_col > t_row) {
            sum = 0.0f;  // Masked position
        } else {
            sum = fmaxf(sum, 0.0f);  // ReLU
        }
        
        att[att_offset + t_row * T + t_col] = sum;
    }
}

__global__ void attention_matmul_v_kernel_ori(
    const float* __restrict__ att,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int B, const int H, const int T, const int hs
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t >= T) return;
    
    const int att_offset = b * H * T * T + h * T * T;
    const int v_offset = b * H * T * hs + h * T * hs;
    const int out_offset = b * H * T * hs + h * T * hs;
    
    for (int i = 0; i < hs; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < T; ++j) {
            sum += att[att_offset + t * T + j] * v[v_offset + j * hs + i];
        }
        out[out_offset + t * hs + i] = sum;
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
    int B = in_batch;
    int H = in_height;
    int T = in_channels;
    int hs = in_width;
    float scale = 1.0f / sqrtf((float)hs);
    
    const int threads = 256;
    const int blocks_t = (T + threads - 1) / threads;
    dim3 blocks(blocks_t, H, B);
    
    fused_qk_mask_relu_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        input + B * H * T * hs,
        output,
        scale, B, H, T, hs
    );
    
    attention_matmul_v_kernel_ori<<<blocks, threads, 0, stream>>>(
        output,
        input + 2 * B * H * T * hs,
        output,
        B