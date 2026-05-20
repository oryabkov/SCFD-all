
ifndef CUDA_ROOT_PATH
CUDA_ROOT_PATH = /usr/local/cuda
endif
ifndef MPI_ROOT_PATH
MPI_ROOT_PATH = /usr/local/mpi
endif

SCFD_INCLUDE=../../include
NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
MPICC = $(MPI_ROOT_PATH)/bin/mpic++
MPIRUN = $(MPI_ROOT_PATH)/bin/mpiexec
HIPCC ?= hipcc
SYCL_CXX ?= icpx
SYCL_CPPSTD ?= -std=c++17
SYCL_TARGET_FLAGS ?=
LIBPATH = -ldl -L$(CUDA_ROOT_PATH)/lib64 -L${MPI_ROOT_PATH}/lib
CUDALIBRARIES = -lcudart
INCLUDE_MPI = -I${MPI_ROOT_PATH}/include
MPICOMPILER_OPENMP = -fopenmp -lpthread
MPI_CXXFLAGS = -std=c++14
MPI_TEST_HOST = localhost:4
MPI_TEST_PROCS = 4
MPI_TEST_RUN_FLAGS = --bind-to none
CUDA_TEST_TARGETS = test_mpi_rect_distributor.bin test_mpi_rect_distributor_tensor.bin
MPI_TEST_TARGETS = test_sequential_mpi.bin test_mpi_binary_file.bin test_mpi_thread_multiple.bin test_mpi_comm_split.bin test_mpi_comm_collectives.bin test_mpi_comm_p2p.bin test_mpi_comm_calls.bin test_backend_runtime_mpi.bin
TEST_TARGETS = $(CUDA_TEST_TARGETS) $(MPI_TEST_TARGETS)
BACKEND_RUNTIME_MPI_DEVICE_TEST = test_backend_runtime_mpi_device.cpp

.PHONY: all clean

all: $(TEST_TARGETS)

test_mpi_rect_distributor.bin: test_mpi_rect_distributor.cu
	$(NVCC) -I${SCFD_INCLUDE} $(INCLUDE_MPI) -O3 test_mpi_rect_distributor.cu -c -o test_mpi_rect_distributor.o
	${MPICC} test_mpi_rect_distributor.o $(LIBPATH) $(CUDALIBRARIES) $(MPICOMPILER_OPENMP) -o test_mpi_rect_distributor.bin
test_mpi_rect_distributor_tensor.bin: test_mpi_rect_distributor_tensor.cu
	$(NVCC) -I${SCFD_INCLUDE} $(INCLUDE_MPI) -O3 test_mpi_rect_distributor_tensor.cu -c -o test_mpi_rect_distributor_tensor.o
	${MPICC} test_mpi_rect_distributor_tensor.o $(LIBPATH) $(CUDALIBRARIES) $(MPICOMPILER_OPENMP) -o test_mpi_rect_distributor_tensor.bin

test_sequential_mpi.bin: test_sequential_mpi.cpp
	${MPICC} -I${SCFD_INCLUDE} test_sequential_mpi.cpp -o test_sequential_mpi.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:77 -n 77 ./test_sequential_mpi.bin

test_mpi_binary_file.bin: test_mpi_binary_file.cpp
	${MPICC} -I${SCFD_INCLUDE} test_mpi_binary_file.cpp -o test_mpi_binary_file.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:77 -n 77 ./test_mpi_binary_file.bin


test_mpi_thread_multiple.bin: test_mpi_thread_multiple.cpp
	${MPICC} -I${SCFD_INCLUDE} test_mpi_thread_multiple.cpp -o test_mpi_thread_multiple.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_thread_multiple.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_thread_multiple.bin 0
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_thread_multiple.bin 1
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_thread_multiple.bin 2
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_thread_multiple.bin 3

	
test_mpi_comm_split.bin: test_mpi_comm_split.cpp
	${MPICC} -I${SCFD_INCLUDE} test_mpi_comm_split.cpp -o test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 2 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 3 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 4 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 5 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 6 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 7 ./test_mpi_comm_split.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host localhost:8 -n 8 ./test_mpi_comm_split.bin

test_mpi_comm_collectives.bin: test_mpi_comm_collectives.cpp
	${MPICC} $(MPI_CXXFLAGS) -I${SCFD_INCLUDE} $(INCLUDE_MPI) test_mpi_comm_collectives.cpp -o test_mpi_comm_collectives.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_mpi_comm_collectives.bin

test_mpi_comm_p2p.bin: test_mpi_comm_p2p.cpp
	${MPICC} $(MPI_CXXFLAGS) -I${SCFD_INCLUDE} $(INCLUDE_MPI) test_mpi_comm_p2p.cpp -o test_mpi_comm_p2p.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_mpi_comm_p2p.bin

test_mpi_comm_calls.bin: test_mpi_comm_calls.cpp
	${MPICC} $(MPI_CXXFLAGS) -I${SCFD_INCLUDE} $(INCLUDE_MPI) test_mpi_comm_calls.cpp -o test_mpi_comm_calls.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_mpi_comm_calls.bin

test_backend_runtime_mpi.bin: test_backend_runtime_mpi.cpp
	${MPICC} $(MPI_CXXFLAGS) -I${SCFD_INCLUDE} $(INCLUDE_MPI) test_backend_runtime_mpi.cpp -o test_backend_runtime_mpi.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_backend_runtime_mpi.bin

test_backend_runtime_mpi_cuda.bin: $(BACKEND_RUNTIME_MPI_DEVICE_TEST)
	$(NVCC) -DPLATFORM_CUDA -DSCFD_BACKEND_ENABLE_MPI -x cu -I${SCFD_INCLUDE} $(INCLUDE_MPI) -O3 $(BACKEND_RUNTIME_MPI_DEVICE_TEST) -c -o test_backend_runtime_mpi_cuda.o
	${MPICC} test_backend_runtime_mpi_cuda.o $(LIBPATH) $(CUDALIBRARIES) $(MPICOMPILER_OPENMP) -o test_backend_runtime_mpi_cuda.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_backend_runtime_mpi_cuda.bin

test_backend_runtime_mpi_hip.bin: $(BACKEND_RUNTIME_MPI_DEVICE_TEST)
	$(HIPCC) $(MPI_CXXFLAGS) -DPLATFORM_HIP -DSCFD_BACKEND_ENABLE_MPI -I${SCFD_INCLUDE} $(INCLUDE_MPI) $(BACKEND_RUNTIME_MPI_DEVICE_TEST) -L${MPI_ROOT_PATH}/lib -lmpi -o test_backend_runtime_mpi_hip.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_backend_runtime_mpi_hip.bin

test_backend_runtime_mpi_sycl.bin: $(BACKEND_RUNTIME_MPI_DEVICE_TEST)
	$(SYCL_CXX) $(SYCL_CPPSTD) $(SYCL_TARGET_FLAGS) -DPLATFORM_SYCL -DSCFD_BACKEND_ENABLE_MPI -fsycl -I${SCFD_INCLUDE} $(INCLUDE_MPI) $(BACKEND_RUNTIME_MPI_DEVICE_TEST) -L${MPI_ROOT_PATH}/lib -lmpi -o test_backend_runtime_mpi_sycl.bin
	$(MPIRUN) $(MPI_TEST_RUN_FLAGS) --host $(MPI_TEST_HOST) -n $(MPI_TEST_PROCS) ./test_backend_runtime_mpi_sycl.bin

clean:
	rm -f *.bin *.o test_file_write.txt
