// don't need to include any header file !!!!

void roi_align_forward_cuda(THCudaTensor *input, THCudaTensor *rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, THCudaTensor *out, THCudaTensor *max_idx, THCudaTensor *max_idy);

void roi_align_backward_cuda(THCudaTensor *in_grad, THCudaTensor *out_grad, THCudaTensor *rois, THCudaTensor *max_idx,
                            THCudaTensor *max_idy, const float spatial_scale);
