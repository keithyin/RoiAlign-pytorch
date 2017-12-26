all : src/cuda/roi_align_kernel.cu.o
	python build.py

src/cuda/roi_align_kernel.cu.o : src/cuda/roi_align_kernel.h src/cuda/roi_align_kernel.cu
	nvcc  -c  -o src/cuda/roi_align_kernel.cu.o src/cuda/roi_align_kernel.cu --compiler-options -fPIC

clean :
	rm -r _ext/
	rm src/cuda/roi_align_kernel.cu.o
