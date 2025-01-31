$(info for cuda target run 'ARCH=XX make cuda')

LIST = float double
template:
	for i in $(LIST); do \
		for number in 2 3 4 5 ; do \
		    echo $$number ; \
			$(CC) -O3 $(FILE) $(OPENMP) -I../../include -DN_NUM=$$number -DREAL=$$i -o a$$i$$number.bin ; \
	    done ; \
	done
info_cuda:
	echo 


template1:
	$(CC) -O3 $(FILE) $(OPENMP) -I../../include -DN_NUM=5 -DREAL=float -o a5number.bin

cuda: 	CC = nvcc -arch=sm_$(ARCH)
cuda:	FILE = test_matmul_cuda.cu
cuda:   OPENMP = -Xcompiler -fopenmp -lpthread
cuda:	TARGET = cuda
cuda:	template

hip:	CC = HIPCC_VERBOSE=7 hipcc
hip:	FILE = test_matmul_hip.cpp
hip:	OPENMP = -fopenmp -lpthread
hip:	TARGET = hip
hip:	template


syclamd:	CC = icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1102
syclamd:	FILE = test_matmul_sycl.cpp
syclamd:	OPENMP = -qopenmp -lpthread
syclamd:	TARGET = sycl
syclamd:	template


syclnv:	CC = icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_$(ARCH)
syclnv:	FILE = test_matmul_sycl.cpp
syclnv:	OPENMP = -qopenmp -lpthread
syclnv: TARGET = sycl
syclnv:	template
