
ifndef CUDA_ROOT_PATH
CUDA_ROOT_PATH = /usr/local/cuda
endif
ifndef MPI_ROOT_PATH
MPI_ROOT_PATH = /usr/local/mpi
endif

SCFD_INCLUDE=../../include
NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
MPICC = $(MPI_ROOT_PATH)/bin/mpic++
LIBPATH = -ldl -L$(CUDA_ROOT_PATH)/lib64 -L${MPI_ROOT_PATH}/lib
CUDALIBRARIES = -lcudart
INCLUDE_MPI = -I${MPI_ROOT_PATH}/include
MPICOMPILER_OPENMP = -fopenmp -lpthread

test_mpi_rect_distributor.bin: test_mpi_rect_distributor.cu
	$(NVCC) -I${SCFD_INCLUDE} $(INCLUDE_MPI) -O3 test_mpi_rect_distributor.cu -c -o test_mpi_rect_distributor.o
	${MPICC} test_mpi_rect_distributor.o $(LIBPATH) $(CUDALIBRARIES) $(MPICOMPILER_OPENMP) -o test_mpi_rect_distributor.bin
test_mpi_rect_distributor_tensor.bin: test_mpi_rect_distributor_tensor.cu
	$(NVCC) -I${SCFD_INCLUDE} $(INCLUDE_MPI) -O3 test_mpi_rect_distributor_tensor.cu -c -o test_mpi_rect_distributor_tensor.o
	${MPICC} test_mpi_rect_distributor_tensor.o $(LIBPATH) $(CUDALIBRARIES) $(MPICOMPILER_OPENMP) -o test_mpi_rect_distributor_tensor.bin

test_sequential_mpi.bin: test_sequential_mpi.cpp
	${MPICC} -I${SCFD_INCLUDE} test_sequential_mpi.cpp -o test_sequential_mpi.bin
	mpiexec --host localhost:77 -n 77 ./test_sequential_mpi.bin

test_mpi_binary_file.bin: test_mpi_binary_file.cpp
	${MPICC} -I${SCFD_INCLUDE} test_mpi_binary_file.cpp -o test_mpi_binary_file.bin
	mpiexec --host localhost:77 -n 77 ./test_mpi_binary_file.bin		