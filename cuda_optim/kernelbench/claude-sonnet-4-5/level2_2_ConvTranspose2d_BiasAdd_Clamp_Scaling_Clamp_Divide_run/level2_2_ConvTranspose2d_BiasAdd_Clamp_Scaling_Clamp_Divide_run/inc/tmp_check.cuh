__global__ void fused_bias_clamp_scale_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        int bias_idx = c;
        
        float val = input[idx] + bias[bias_idx];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val / scaling_factor;
        
        output[idx] = val;
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
    const float scaling_factor = 2.0f;
    const int block_size = 256;
    int total_elements = in_elems;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_clamp_scale_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(input) + in_elems, // placeholder: actual bias ptr not passed; see note below
        reinterpret_cast<float*>(output),
        in_batch,
        in_channels,
        in_height,
        in_width,
        scaling_factor
    );
}