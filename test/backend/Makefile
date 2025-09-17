ifndef CUDA_ROOT_PATH
CUDA_ROOT_PATH = /usr/local/cuda
endif

SCFD_INCLUDE=../../include
CC = g++
NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
LIBCUDAPATH = -ldl -L$(CUDA_ROOT_PATH)/lib64
LIBHIPPATH = 
LIBSYCLPATH = 
OPENMP = -fopenmp -lpthread 


test_backend_serial_cpu.bin: test_backend_serial_cpu.cpp
	$(CC) test_backend_serial_cpu.cpp -I$(SCFD_INCLUDE) -o test_backend_serial_cpu.bin
test_backend_types_cuda.bin: test_backend_types_cuda.cu
	$(NVCC) test_backend_types_cuda.cu -I$(SCFD_INCLUDE) -o test_backend_types_cuda.bin
test_backend_omp_cpu.bin: test_backend_omp_cpu.cpp
	$(CC) test_backend_omp_cpu.cpp -I$(SCFD_INCLUDE) $(OPENMP) -o test_backend_omp_cpu.bin
