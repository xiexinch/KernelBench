__global__ void fused_ops_kernel_opt(
    const float* input,
    const float* scaling_factor,
    const float* bias,
    float* output,
    int size,
    int out_channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int channel = (idx / spatial_size) % out_channels;
        float val = input[idx];
        val = val * scaling_factor[channel];
        val = tanhf(val);
        val = val * bias[channel];
        val = 1.0f / (1.0f + expf(-val));
        output[idx] = val;
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
    int size = in_elems;
    // Assuming input is 5D: [batch, channels, depth, height, width]
    // From the original code: spatial_size = shape[2] * shape[3] * shape[4]
    // So we need to infer depth from total elements:
    // in_elems = in_batch * in_channels * depth * in_height * in_width
    // But the function signature doesn't provide depth explicitly.
    // However, from the original usage, we know:
    //   out_channels = in_channels (after conv, but here passed as argument)
    //   spatial_size = in_elems / (in_batch * out_channels)
    int spatial_size = in_elems / (in_batch * out_channels);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_ops_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(input + in_elems),      // scaling_factor assumed right after input
        reinterpret_cast<const float*>(input + in_elems + out_channels), // bias assumed right after scaling_factor
        reinterpret_cast<float*>(output),
        size,
        out_channels,
        spatial_size
    );
}