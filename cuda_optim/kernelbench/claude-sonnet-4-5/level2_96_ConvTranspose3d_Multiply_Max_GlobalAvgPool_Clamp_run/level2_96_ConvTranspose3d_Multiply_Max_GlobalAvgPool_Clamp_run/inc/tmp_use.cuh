__global__ void scale_maxpool3d_kernel_opt(
    const float* input,
    float* output,
    float scale,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    
    if (idx < total_elements) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int d_out = (idx / (out_w * out_h)) % out_d;
        int c = (idx / (out_w * out_h * out_d)) % channels;
        int b = idx / (out_w * out_h * out_d * channels);
        
        float max_val = -INFINITY;
        
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out * kernel_size + kd;
                    int h_in = h_out * kernel_size + kh;
                    int w_in = w_out * kernel_size + kw;
                    
                    if (d_in < in_d && h_in < in_h && w_in < in_w) {
                        int in_idx = b * (channels * in_d * in_h * in_w) +
                                    c * (in_d * in_h * in_w) +
                                    d_in * (in_h * in_w) +
                                    h_in * in_w +
                                    w_in;
                        float val = input[in_idx] * scale;
                        max_val = fmaxf(max_val, val);
                    }
                }
            }
        }
        
        output[idx] = max_val;
    }
}

__global__ void global_avgpool_clamp_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int d, int h, int w,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = batch_size * channels;
    
    if (idx < total_channels) {
        int spatial_size = d * h * w;
        int offset = idx * spatial_size;
        
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input[offset + i];
        }
        
        float avg = sum / spatial_size;
        
        // Clamp
        avg = fminf(fmaxf(avg, clamp_min), clamp_max);
        
        output[idx] = avg;
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
    // Determine which kernel to launch based on output dimensions
    // If output spatial dims are 1x1x1, assume global_avgpool_clamp
    // Otherwise, assume scale_maxpool3d

    const int block_size = 256;
    float scale = 0.5f;
    int kernel_size = 2;
    float clamp_min = 0.0f;
    float clamp_max = 1.0f;

    // Extract 3D dimensions from given parameters
    // Input: [batch, channels, D, H, W]
    // Given: in_batch, in_channels, in_height=H, in_width=W
    // We need to infer D from in_elems: in_elems = in_batch * in_channels * D * in_height * in_width
    int in_d = (in_batch > 0 && in_channels > 0 && in_height > 0 && in_width > 0) 
               ? (in_elems / (in_batch * in_channels * in_height * in_width)) : 1;
    int in_h = in_height;
    int in_w = in_width;

    int out_d = (out_batch > 0 && out_channels > 0 && out_height > 0 && out_width > 0) 
                ? (out_elems / (out_batch * out_channels * out_height * out_width)) : 1;
    int out_h = out_height;
    int out_w = out_width;

    if (out_d == 1 && out_h == 1 && out_w == 1) {
        // Launch global_avgpool_clamp_kernel_opt
        int total_channels = out_batch * out_channels;
        int num_blocks = (total_channels + block_size - 1) / block_size;
        global_avgpool_clamp_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            out_batch, out_channels,
            in_d, in_h, in_w,
            clamp_min, clamp_max
        );
    } else {
        // Launch scale_maxpool3d_kernel_opt
        int total_elements = out_elems;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        scale_maxpool3d_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output),
            scale,
            in_batch, in_channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_size
        );
    }
}