from torch import autograd
from _ext import roi_align


class RoiAlignFunction(autograd.Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoiAlignFunction, self).__init__()
        self._pooled_height = int(pooled_height)
        self._pooled_width = int(pooled_width)
        self._spatial_scale = float(spatial_scale)

    def forward(self, x, rois):
        assert len(x.size()) == 4
        assert len(rois.size()) == 2
        assert rois.size(1) == 5

        num_rois = rois.size(0)
        depth = x.size(1)
        argmax_x = x.new().resize_(num_rois, depth, self._pooled_height, self._pooled_width).zero_()
        argmax_y = x.new().resize_(num_rois, depth, self._pooled_height, self._pooled_width).zero_()

        output = x.new(num_rois, depth, self._pooled_height, self._pooled_width).zero_()

        if x.is_cuda:
            roi_align.roi_align_forward_cuda(x, rois, self._pooled_height, self._pooled_width,
                                             self._spatial_scale, output, argmax_x, argmax_y)
        else:
            raise NotImplementedError

        self.save_for_backward(x, rois, argmax_x, argmax_y)

        return output, rois, argmax_x, argmax_y

    def backward(self, grad_ouput, rois_grad, argmax_x_grad, argmax_y_grad):
        x, rois, argmax_x, argmax_y = self.saved_tensors
        grad_input = x.new().resize_as_(x).zero_()

        if grad_ouput.is_cuda:
            roi_align.roi_align_backward_cuda(grad_input, grad_ouput, rois, argmax_x, argmax_y, self._spatial_scale)
        else:
            raise NotImplementedError
        return grad_input, None


def roi_align_op(x, rois, pooled_size, spatial_scale):
    if isinstance(pooled_size, int):
        pooled_height = pooled_width = pooled_size
    else:
        pooled_height, pooled_width = pooled_size
    operation = RoiAlignFunction(pooled_height, pooled_width, spatial_scale)
    out, roi_, argmax_x, argmax_y = operation(x, rois)

    return out
