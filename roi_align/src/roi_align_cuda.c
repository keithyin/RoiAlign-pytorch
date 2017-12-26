#include "THC/THC.h"
#include "./cuda/roi_align_kernel.h"

extern THCState *state;





void roi_align_forward_cuda(THCudaTensor *input, THCudaTensor *rois,
                            int pooled_height, int pooled_width,
                            float spatial_scale, THCudaTensor *out, THCudaTensor *max_idx,
                            THCudaTensor *max_idy){
    // do a lot of check in python backend

//    const int batch_size = input->size[0];
    const int in_channels = input->size[1];
    const int num_rois = rois->size[0];
    const int feat_height = input->size[2];
    const int feat_width = input->size[3];
    THAssert(THCudaTensor_isContiguous(state, input));
    // init out tensor

    THCudaTensor_resize4d(state, out, num_rois, in_channels, pooled_height, pooled_width);

    const float *input_data = THCudaTensor_data(state, input);
    const float *rois_data = THCudaTensor_data(state, rois);
    float *argmax_x = THCudaTensor_data(state, max_idx);
    float *argmax_y = THCudaTensor_data(state, max_idy);
    float *out_data = THCudaTensor_data(state, out);

    // prepare cuda kernel invoke
    const int total_count = num_rois * pooled_height * pooled_width * in_channels;

    cudaStream_t cur_stream = THCState_getCurrentStream(state);
    roi_align_forward_cuda_launcher(total_count, input_data, spatial_scale, in_channels, feat_height, feat_width,
                                    pooled_height, pooled_width, rois_data, out_data, argmax_x, argmax_y,
                                    cur_stream);

}


void roi_align_backward_cuda(THCudaTensor *in_grad, THCudaTensor *out_grad, THCudaTensor *rois,
                             THCudaTensor *max_idx,
                             THCudaTensor *max_idy, const float spatial_scale) {
    const int batch_size = in_grad->size[0];
    const int depth = in_grad->size[1];
    const int feat_height = in_grad->size[2];
    const int feat_width = in_grad->size[3];

    const int total_count = batch_size * depth * feat_height * feat_width;

    const int num_rois = rois->size[0];
    const int pooled_height = out_grad->size[2];
    const int pooled_width = out_grad->size[3];


    float *in_grad_data = THCudaTensor_data(state, in_grad);
    const float *rois_data = THCudaTensor_data(state, rois);
    const float *out_grad_data = THCudaTensor_data(state, out_grad);
    const float *argmax_x = THCudaTensor_data(state, max_idx);
    const float *argmax_y = THCudaTensor_data(state, max_idy);

    cudaStream_t cur_stream = THCState_getCurrentStream(state);

    roi_align_backward_cuda_launcher(
            total_count, out_grad_data, argmax_x, argmax_y, num_rois, spatial_scale, depth, feat_height, feat_width,
            pooled_height, pooled_width, in_grad_data, rois_data, cur_stream);


}