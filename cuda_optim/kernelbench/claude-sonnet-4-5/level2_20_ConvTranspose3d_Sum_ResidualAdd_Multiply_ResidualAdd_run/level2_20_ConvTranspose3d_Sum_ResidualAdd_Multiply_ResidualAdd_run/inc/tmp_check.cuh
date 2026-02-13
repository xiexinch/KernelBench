__global__ void fused_residual_ops_kernel_ori(
    const float* conv_out,
    const float* bias,
    float* output,
    int total_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float conv_val = conv_out[idx];
        float bias_val = bias[c];
        
        // x = conv_out + bias
        float x = conv_val + bias_val;
        // x = x + conv_out
        x = x + conv_val;
        // x = x * conv_out
        x = x * conv_val;
        // x = x + conv_out
        x = x + conv_val;
        
        output[idx] = x;
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
    int total_size = in_elems;
    int channels = in_channels;
    int spatial_size = in_height * in_width; // assumes 3D input layout flattened as B*C*D*H*W -> spatial = D*H*W, but here we only have H and W passed.
    // Note: Original code used depth*height*width. Since depth isn't passed separately, we assume it's folded into in_height.
    // This matches the typical flattening: total_size = batch * channels * spatial, and spatial = total_size / (batch * channels)

    // Recompute spatial_size robustly from total_size if needed:
    // But per interface contract, we rely on passed dims. Given lack of depth, we treat in_height as containing depth*height.
    // So spatial_size = in_elems / (in_batch * in_channels)
    spatial_size = in_elems / (in_batch * in_channels);

    const int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;

    fused_residual_ops_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output), // Note: bias is expected in 'output' param due to interface limitation
        reinterpret_cast<float*>(output),
        total_size,
        channels,
        spatial_size
    );
}