#CUDA_ARCHITECTURE = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70
CUDA_ARCHITECTURE = 
HIP_ARCHITECTURE = 
INCLUDE_SRC = ./src
INCLUDE_SCFD = ../../include
ROOT_CUDA = /usr/local/cuda
ROOT_HIP = /opt/rocm
ROOT_MPI = /usr/local/mpi
INCLUDE_CUDA = $(ROOT_CUDA)/include
INCLUDE_HIP = $(ROOT_HIP)/include
LIB_CUDA = $(ROOT_CUDA)/lib64
LIB_HIP = $(ROOT_HIP)/lib
INCLUDE_MPI = $(ROOT_MPI)/include
LIB_MPI = $(ROOT_MPI)/lib
NVCC = $(ROOT_CUDA)/bin/nvcc
HIPCC = $(ROOT_HIP)/bin/hipcc
MPICC = $(ROOT_MPI)/bin/mpic++
OPENMP = -fopenmp -lpthread

cuda:
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCHITECTURE) -I$(INCLUDE_SCFD) -I$(INCLUDE_SRC) -I$(INCLUDE_CUDA) -I$(INCLUDE_MPI) -DCUDA src/mgpu_reduce.cpp -c -o mgpu_reduce.obj -Xcompiler -fopenmp
	$(MPICC) mgpu_reduce.obj -L$(LIB_CUDA) -L$(LIB_MPI) -lcudart -o mgpu_reduce.bin $(OPENMP)
	rm mgpu_reduce.obj

hip:
	$(HIPCC) -O3 -std=c++17 $(HIP_ARCHITECTURE) -I$(INCLUDE_SCFD) -I$(INCLUDE_SRC) -I$(INCLUDE_HIP) -I$(INCLUDE_MPI) -DHIP src/mgpu_reduce.cpp -c -o mgpu_reduce.obj -fopenmp -fPIE
	$(HIPCC) mgpu_reduce.obj -L$(LIB_HIP) -L$(LIB_MPI) -lmpi  -o mgpu_reduce.bin $(OPENMP) 
	rm mgpu_reduce.obj