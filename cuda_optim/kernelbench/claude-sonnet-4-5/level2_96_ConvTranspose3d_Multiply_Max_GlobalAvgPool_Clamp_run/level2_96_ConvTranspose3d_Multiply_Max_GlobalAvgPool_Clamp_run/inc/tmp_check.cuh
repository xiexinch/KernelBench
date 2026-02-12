__global__ void scale_maxpool3d_kernel_ori(
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

__global__ void global_avgpool_clamp_kernel_ori(
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // First kernel: scale_maxpool3d
    int batch_size = in_batch;
    int channels = in_channels;
    int in_d = in_height;
    int in_h = in_channels;
    int in_w = in_width;
    float scale = 0.5;
    int kernel_size = 2;
    
    int out_d = in_d / kernel_size;
    int out_h = in_h /