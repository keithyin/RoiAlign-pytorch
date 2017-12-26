from torch import nn
from ..functions import roi_align


class RoiAlign(nn.Module):
    def __init__(self, pooled_size, spatial_scale):
        """spatial_scale : 1./total_stride"""
        super(RoiAlign, self).__init__()
        if isinstance(pooled_size, int):
            pooled_height = pooled_width = pooled_size
        else:
            pooled_height, pooled_width = pooled_size
        self._pooled_height = pooled_height
        self._pooled_width = pooled_width
        self._spatial_scale = spatial_scale

    def forward(self, x, rois):
        """x: Variable: FloatTensor [bs, depth, height, width]
           rois:Variable:FloatTensor [num_rois, 5], 5 indicate [bs_idx, x1, y1, x2, y2]
           [x1, y1, x2, y2] indicate the coordinate in the input image.
        """
        res = roi_align.roi_align_op(x, rois, (self._pooled_height, self._pooled_width), self._spatial_scale)
        return res
