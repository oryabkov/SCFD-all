ifndef CUDA_ROOT_PATH
CUDA_ROOT_PATH = /usr/local/cuda
endif

SCFD_INCLUDE=../../include
CC = g++
NVCC = $(CUDA_ROOT_PATH)/bin/nvcc
CXXFLAGS ?= -std=c++14
CUDAFLAGS ?= -std=c++14
PERF_CXXFLAGS ?= $(CXXFLAGS) -O3
PERF_CUDAFLAGS ?= $(CUDAFLAGS) -O3
HIPCC ?= hipcc
SYCL_CXX ?= icpx
LIBCUDAPATH = -ldl -L$(CUDA_ROOT_PATH)/lib64
LIBHIPPATH =
LIBSYCLPATH =
OPENMP = -fopenmp -lpthread 
PERF_SIZE ?= 8388608
PERF_REPEATS ?= 3
PERF_MIN_SPEEDUP ?= 1.05
CPU_BINS = test_backend_serial_cpu.bin test_backend_omp_cpu.bin test_backend_algorithms_serial_cpu.bin test_backend_algorithms_omp_cpu.bin test_backend_performance_serial_cpu.bin test_backend_performance_omp_cpu.bin
CUDA_BINS = test_backend_types_cuda.bin test_backend_algorithms_cuda.bin test_backend_performance_cuda.bin
HIP_BINS = test_backend_types_hip.bin test_backend_algorithms_hip.bin test_backend_performance_hip.bin
SYCL_BINS = test_backend_types_sycl.bin test_backend_algorithms_sycl.bin test_backend_performance_sycl.bin

.PHONY: all cpu cuda hip sycl perf_cpu perf_cuda perf_hip perf_sycl run_cpu run_cuda run_hip run_sycl run_perf_cpu run_perf_cuda run_perf_hip run_perf_sycl clean

all: cpu cuda

cpu: $(CPU_BINS)

cuda: $(CUDA_BINS)

hip: $(HIP_BINS)

sycl: $(SYCL_BINS)

perf_cpu: test_backend_performance_serial_cpu.bin test_backend_performance_omp_cpu.bin

perf_cuda: test_backend_performance_cuda.bin

perf_hip: test_backend_performance_hip.bin

perf_sycl: test_backend_performance_sycl.bin


test_backend_serial_cpu.bin: test_backend_serial_cpu.cpp
	$(CC) $(CXXFLAGS) test_backend_serial_cpu.cpp -I$(SCFD_INCLUDE) -o test_backend_serial_cpu.bin
test_backend_types_cuda.bin: test_backend_types_cuda.cu
	$(NVCC) $(CUDAFLAGS) test_backend_types_cuda.cu -I$(SCFD_INCLUDE) -o test_backend_types_cuda.bin
test_backend_omp_cpu.bin: test_backend_omp_cpu.cpp
	$(CC) $(CXXFLAGS) test_backend_omp_cpu.cpp -I$(SCFD_INCLUDE) $(OPENMP) -o test_backend_omp_cpu.bin

test_backend_algorithms_serial_cpu.bin: test_backend_algorithms_serial_cpu.cpp test_backend_algorithms_common.h
	$(CC) $(CXXFLAGS) test_backend_algorithms_serial_cpu.cpp -I$(SCFD_INCLUDE) -o test_backend_algorithms_serial_cpu.bin

test_backend_algorithms_omp_cpu.bin: test_backend_algorithms_omp_cpu.cpp test_backend_algorithms_common.h
	$(CC) $(CXXFLAGS) test_backend_algorithms_omp_cpu.cpp -I$(SCFD_INCLUDE) $(OPENMP) -o test_backend_algorithms_omp_cpu.bin

test_backend_algorithms_cuda.bin: test_backend_algorithms_cuda.cu test_backend_algorithms_common.h
	$(NVCC) $(CUDAFLAGS) test_backend_algorithms_cuda.cu -I$(SCFD_INCLUDE) -o test_backend_algorithms_cuda.bin

test_backend_types_hip.bin: test_backend_types_hip.cpp
	$(HIPCC) $(CXXFLAGS) test_backend_types_hip.cpp -I$(SCFD_INCLUDE) -o test_backend_types_hip.bin

test_backend_algorithms_hip.bin: test_backend_algorithms_hip.cpp test_backend_algorithms_common.h
	$(HIPCC) $(CXXFLAGS) test_backend_algorithms_hip.cpp -I$(SCFD_INCLUDE) -o test_backend_algorithms_hip.bin

test_backend_algorithms_sycl.bin: test_backend_algorithms_sycl.cpp test_backend_algorithms_common.h
	$(SYCL_CXX) $(CXXFLAGS) -fsycl test_backend_algorithms_sycl.cpp -I$(SCFD_INCLUDE) -o test_backend_algorithms_sycl.bin

test_backend_types_sycl.bin: test_backend_types_sycl.cpp
	$(SYCL_CXX) $(CXXFLAGS) -fsycl test_backend_types_sycl.cpp -I$(SCFD_INCLUDE) -o test_backend_types_sycl.bin

test_backend_performance_serial_cpu.bin: test_backend_performance_serial_cpu.cpp test_backend_performance_common.h
	$(CC) $(PERF_CXXFLAGS) test_backend_performance_serial_cpu.cpp -I$(SCFD_INCLUDE) -o test_backend_performance_serial_cpu.bin

test_backend_performance_omp_cpu.bin: test_backend_performance_omp_cpu.cpp test_backend_performance_common.h
	$(CC) $(PERF_CXXFLAGS) test_backend_performance_omp_cpu.cpp -I$(SCFD_INCLUDE) $(OPENMP) -o test_backend_performance_omp_cpu.bin

test_backend_performance_cuda.bin: test_backend_performance_cuda.cu test_backend_performance_common.h
	$(NVCC) $(PERF_CUDAFLAGS) test_backend_performance_cuda.cu -I$(SCFD_INCLUDE) -o test_backend_performance_cuda.bin

test_backend_performance_hip.bin: test_backend_performance_hip.cpp test_backend_performance_common.h
	$(HIPCC) $(PERF_CXXFLAGS) test_backend_performance_hip.cpp -I$(SCFD_INCLUDE) -o test_backend_performance_hip.bin

test_backend_performance_sycl.bin: test_backend_performance_sycl.cpp test_backend_performance_common.h
	$(SYCL_CXX) $(PERF_CXXFLAGS) -fsycl test_backend_performance_sycl.cpp -I$(SCFD_INCLUDE) -o test_backend_performance_sycl.bin

run_cpu: cpu
	./test_backend_serial_cpu.bin
	./test_backend_omp_cpu.bin
	./test_backend_algorithms_serial_cpu.bin
	./test_backend_algorithms_omp_cpu.bin

run_cuda: cuda
	./test_backend_types_cuda.bin
	./test_backend_algorithms_cuda.bin

run_hip: hip
	./test_backend_types_hip.bin
	./test_backend_algorithms_hip.bin

run_sycl: sycl
	./test_backend_types_sycl.bin
	./test_backend_algorithms_sycl.bin

run_perf_cpu: perf_cpu
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) ./test_backend_performance_serial_cpu.bin
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) ./test_backend_performance_omp_cpu.bin

run_perf_cuda: perf_cuda
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) ./test_backend_performance_cuda.bin

run_perf_hip: perf_hip
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) ./test_backend_performance_hip.bin

run_perf_sycl: perf_sycl
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) ./test_backend_performance_sycl.bin

clean:
	rm -f *.bin
