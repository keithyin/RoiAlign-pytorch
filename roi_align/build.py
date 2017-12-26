import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_align_cuda.c']
    headers += ['src/roi_align_cuda.h']
    # headers += ['src/cuda/roi_align_kernel.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
else:
    raise ValueError("only cuda version support")

extra_objects = ['src/cuda/roi_align_kernel.cu.o']

# absolute path
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.roi_align',  # _ext/roi_align
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
