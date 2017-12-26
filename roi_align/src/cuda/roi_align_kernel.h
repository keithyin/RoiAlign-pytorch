//
// Created by keith on 12/26/17.
//
void roi_align_forward_cuda_launcher(const int count, const float *bottom_data,
                                     const float spatial_scale,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const float *bottom_rois, float *top_data,
                                     float *argmax_x, float *argmax_y,
                                     cudaStream_t cur_stream);

void roi_align_backward_cuda_launcher(const int count, const float *top_diff,
                                      const float *argmax_x, const float *argmax_y,
                                      const int num_rois,
                                      const float spatial_scale,
                                      const int channels, const int height, const int width,
                                      const int pooled_height, const int pooled_width,
                                      float *bottom_diff, const float *bottom_rois,
                                      cudaStream_t cur_stream);


