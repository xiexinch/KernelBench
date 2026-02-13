__global__ void scale_min_reduce_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scale_factor,
    const int batch_size,
    const int channels,
    const int spatial_size
) {
    int batch_idx = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= spatial_size) return;
    
    float min_val = INFINITY;
    
    // Iterate through all channels for this batch and spatial location
    for (int c = 0; c < channels; c++) {
        int idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        float val = input[idx] * scale_factor;
        min_val = fminf(min_val, val);
    }
    
    // Write output
    int out_idx = batch_idx * spatial_size + spatial_idx;
    output[out_idx] = min_val;
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    const int batch_size = in_batch;
    const int channels = in_channels;
    const int spatial_size = in_height * in_width;
    const float scale_factor = 2.0f;

    const int threads = 256;
    const int blocks_x = (spatial_size + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);

    scale_min_reduce_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        scale_factor,
        batch_size,
        channels,
        spatial_size
    );
}