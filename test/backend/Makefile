CUDA_ROOT_PATH ?= /usr/local/cuda
MPI_ROOT_PATH ?= /usr/local/mpi
SYCL_DPL_ROOT_PATH ?= /opt/intel/oneapi-sycl-nvidia/dpl/2022.7

SCFD_INCLUDE := ../../include

CXX ?= g++
NVCC ?= $(CUDA_ROOT_PATH)/bin/nvcc
HIPCC ?= hipcc
SYCL_CXX ?= $(if $(SYCL_PATH),$(SYCL_PATH)/bin/icpx,icpx)
MPICXX ?= $(MPI_ROOT_PATH)/bin/mpic++
MPIRUN ?= $(MPI_ROOT_PATH)/bin/mpiexec

CPPSTD ?= -std=c++14
SYCL_CPPSTD ?= -std=c++17
CXXFLAGS ?= $(CPPSTD)
CUDAFLAGS ?= $(CPPSTD)
HIPFLAGS ?= $(CPPSTD)
SYCLFLAGS ?= $(SYCL_CPPSTD)
CUDA_LDFLAGS ?=
SYCL_TARGET_FLAGS ?=
SYCL_DPL_FLAGS ?= -I$(SYCL_DPL_ROOT_PATH)/include

PERF_CXXFLAGS ?= $(CXXFLAGS) -O3
PERF_CUDAFLAGS ?= $(CUDAFLAGS) -O3
PERF_HIPFLAGS ?= $(HIPFLAGS) -O3
PERF_SYCLFLAGS ?= $(SYCLFLAGS) -O3

OPENMP_FLAGS ?= -fopenmp -lpthread
CUDA_OPENMP_FLAGS ?= -Xcompiler -fopenmp -lgomp -lpthread
CUDA_OPENMP_COMPILE_FLAGS ?= -Xcompiler -fopenmp
HIP_OPENMP_FLAGS ?= -fopenmp -lpthread
SYCL_OPENMP_FLAGS ?= $(OPENMP_FLAGS)

MPI_CPPFLAGS ?= -I$(MPI_ROOT_PATH)/include
MPI_LDFLAGS ?= -L$(MPI_ROOT_PATH)/lib
MPI_LIBS ?= -lmpi
CUDA_MPI_LIBS ?= -L$(CUDA_ROOT_PATH)/lib64 -lcudart -ldl
MPIEXEC_FLAGS ?= --bind-to none
MPI_PROCS ?= 1

PERF_SIZE ?= 6388608
PERF_ALGO_SIZE ?= $(PERF_SIZE)
PERF_REPEATS ?= 3
PERF_MIN_SPEEDUP ?= 1.05

PLATFORM ?= SERIAL

ifneq ($(strip $(PLATFORM_MPI)),)
ifeq ($(origin MPI),undefined)
MPI := $(PLATFORM_MPI)
else ifneq ($(strip $(MPI)),$(strip $(PLATFORM_MPI)))
$(error MPI=$(MPI) conflicts with PLATFORM_MPI=$(PLATFORM_MPI))
endif
endif
MPI ?= OFF

VALID_PLATFORMS := SERIAL OMP CUDA HIP SYCL
VALID_MPI_VALUES := OFF ON

ifeq ($(filter $(PLATFORM),$(VALID_PLATFORMS)),)
$(error Invalid PLATFORM='$(PLATFORM)'; choose one of: $(VALID_PLATFORMS))
endif
ifeq ($(filter $(MPI),$(VALID_MPI_VALUES)),)
$(error Invalid MPI='$(MPI)'; choose one of: $(VALID_MPI_VALUES))
endif

PLATFORM_TAG_SERIAL := serial
PLATFORM_TAG_OMP := omp
PLATFORM_TAG_CUDA := cuda
PLATFORM_TAG_HIP := hip
PLATFORM_TAG_SYCL := sycl

PLATFORM_DEFINE_SERIAL := -DPLATFORM_SERIAL_CPU
PLATFORM_DEFINE_OMP := -DPLATFORM_OMP
PLATFORM_DEFINE_CUDA := -DPLATFORM_CUDA
PLATFORM_DEFINE_HIP := -DPLATFORM_HIP
PLATFORM_DEFINE_SYCL := -DPLATFORM_SYCL

CONFIG := $(PLATFORM_TAG_$(PLATFORM))
PLATFORM_DEFINE := $(PLATFORM_DEFINE_$(PLATFORM))

ifeq ($(MPI),ON)
CONFIG := $(CONFIG)-mpi
MPI_DEFINE := -DSCFD_BACKEND_ENABLE_MPI
MPI_COMPILE_FLAGS := $(MPI_CPPFLAGS)
MPI_LINK_FLAGS := $(MPI_LDFLAGS) $(MPI_LIBS)
RUNNER := $(MPIRUN) $(MPIEXEC_FLAGS) -n $(MPI_PROCS)
endif

BUILD_DIR := build/$(CONFIG)
COMMON_CPPFLAGS := -I$(SCFD_INCLUDE) $(PLATFORM_DEFINE) $(MPI_DEFINE) $(MPI_COMPILE_FLAGS)

ALGORITHMS_BIN := $(BUILD_DIR)/test_backend_algorithms.bin
RUNTIME_BIN := $(BUILD_DIR)/test_backend_runtime.bin
PERFORMANCE_BIN := $(BUILD_DIR)/test_backend_performance.bin
GENERIC_CORE_BINS := $(ALGORITHMS_BIN) $(RUNTIME_BIN)
CORE_BINS := $(GENERIC_CORE_BINS)

ifneq ($(filter $(PLATFORM),SERIAL OMP),)
HOST_BIN := $(BUILD_DIR)/test_backend_host.bin
GENERIC_CORE_BINS += $(HOST_BIN)
CORE_BINS += $(HOST_BIN)
endif

ifeq ($(PLATFORM),CUDA)
CUDA_SCAN_BIN := $(BUILD_DIR)/test_backend_exclusive_scan_omp_vs_cuda.bin
CORE_BINS += $(CUDA_SCAN_BIN)
endif

$(ALGORITHMS_BIN): test_backend_algorithms.cpp test_backend_algorithms_common.h test_backend_config.h
$(RUNTIME_BIN): test_backend_runtime.cpp test_backend_runtime_common.h test_backend_config.h
$(PERFORMANCE_BIN): test_backend_performance.cpp test_backend_performance_common.h test_backend_config.h
ifneq ($(HOST_BIN),)
$(HOST_BIN): test_backend_host.cpp test_backend_config.h
endif

.PHONY: all run perf run-perf clean help

all: $(CORE_BINS)

perf: $(PERFORMANCE_BIN)

run: all
	$(RUNNER) $(ALGORITHMS_BIN)
	$(RUNNER) $(RUNTIME_BIN)
ifneq ($(HOST_BIN),)
	$(RUNNER) $(HOST_BIN)
endif
ifneq ($(CUDA_SCAN_BIN),)
	$(RUNNER) $(CUDA_SCAN_BIN)
endif

run-perf: perf
	SCFD_BACKEND_PERF_SIZE=$(PERF_SIZE) \
	SCFD_BACKEND_PERF_ALGO_SIZE=$(PERF_ALGO_SIZE) \
	SCFD_BACKEND_PERF_REPEATS=$(PERF_REPEATS) \
	SCFD_BACKEND_PERF_MIN_SPEEDUP=$(PERF_MIN_SPEEDUP) \
	$(RUNNER) $(PERFORMANCE_BIN)

$(BUILD_DIR):
	mkdir -p $@

ifeq ($(PLATFORM),SERIAL)
ifeq ($(MPI),ON)
BACKEND_CXX := $(MPICXX)
else
BACKEND_CXX := $(CXX)
endif
BACKEND_FLAGS := $(CXXFLAGS)
PERFORMANCE_FLAGS := $(PERF_CXXFLAGS) $(OPENMP_FLAGS)
else ifeq ($(PLATFORM),OMP)
ifeq ($(MPI),ON)
BACKEND_CXX := $(MPICXX)
else
BACKEND_CXX := $(CXX)
endif
BACKEND_FLAGS := $(CXXFLAGS) $(OPENMP_FLAGS)
PERFORMANCE_FLAGS := $(PERF_CXXFLAGS) $(OPENMP_FLAGS)
else ifeq ($(PLATFORM),HIP)
BACKEND_CXX := $(HIPCC)
BACKEND_FLAGS := $(HIPFLAGS)
PERFORMANCE_FLAGS := $(PERF_HIPFLAGS) $(HIP_OPENMP_FLAGS)
else ifeq ($(PLATFORM),SYCL)
BACKEND_CXX := $(SYCL_CXX)
BACKEND_FLAGS := $(SYCLFLAGS) $(SYCL_TARGET_FLAGS) $(SYCL_DPL_FLAGS) -fsycl
PERFORMANCE_FLAGS := $(PERF_SYCLFLAGS) $(SYCL_TARGET_FLAGS) $(SYCL_DPL_FLAGS) -fsycl $(SYCL_OPENMP_FLAGS)
endif

ifneq ($(PLATFORM),CUDA)
$(GENERIC_CORE_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(BACKEND_CXX) $(CPPFLAGS) $(BACKEND_FLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@

$(PERFORMANCE_BIN): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(BACKEND_CXX) $(CPPFLAGS) $(PERFORMANCE_FLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@
else ifeq ($(MPI),OFF)
$(GENERIC_CORE_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(COMMON_CPPFLAGS) -x cu $< $(CUDA_LDFLAGS) $(LDLIBS) -o $@

$(PERFORMANCE_BIN): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(PERF_CUDAFLAGS) $(CUDA_OPENMP_FLAGS) $(COMMON_CPPFLAGS) -x cu $< $(CUDA_LDFLAGS) $(LDLIBS) -o $@

$(CUDA_SCAN_BIN): test_backend_exclusive_scan_omp_vs_cuda.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(PERF_CUDAFLAGS) $(CUDA_OPENMP_FLAGS) $(COMMON_CPPFLAGS) $< $(CUDA_LDFLAGS) $(LDLIBS) -o $@
else
$(GENERIC_CORE_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(COMMON_CPPFLAGS) -x cu -c $< -o $(@:.bin=.o)
	$(MPICXX) $(@:.bin=.o) $(MPI_LINK_FLAGS) $(LDFLAGS) $(CUDA_MPI_LIBS) $(LDLIBS) -o $@

$(PERFORMANCE_BIN): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(PERF_CUDAFLAGS) $(CUDA_OPENMP_COMPILE_FLAGS) $(COMMON_CPPFLAGS) -x cu -c $< -o $(@:.bin=.o)
	$(MPICXX) $(@:.bin=.o) $(MPI_LINK_FLAGS) $(LDFLAGS) $(OPENMP_FLAGS) $(CUDA_MPI_LIBS) $(LDLIBS) -o $@

$(CUDA_SCAN_BIN): test_backend_exclusive_scan_omp_vs_cuda.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(PERF_CUDAFLAGS) $(CUDA_OPENMP_COMPILE_FLAGS) $(COMMON_CPPFLAGS) -c $< -o $(@:.bin=.o)
	$(MPICXX) $(@:.bin=.o) $(MPI_LINK_FLAGS) $(LDFLAGS) $(OPENMP_FLAGS) $(CUDA_MPI_LIBS) $(LDLIBS) -o $@
endif

clean:
	$(RM) -r build *.bin *.o

help:
	@echo "SCFD backend tests"
	@echo "  make [all] PLATFORM={SERIAL,OMP,CUDA,HIP,SYCL} MPI={OFF,ON}"
	@echo "  make run      Build and run functional tests"
	@echo "  make perf     Build the performance test"
	@echo "  make run-perf Build and run the performance test"
	@echo "  make clean    Remove configuration build directories"
	@echo ""
	@echo "Defaults: PLATFORM=SERIAL MPI=OFF"
	@echo "PLATFORM_MPI is accepted as an alias for MPI. MPI runs default to one rank; override MPI_PROCS as needed."
	@echo "Run 'make clean' after changing compiler or low-level flag variables."
	@echo "CUDA setting: CUDA_LDFLAGS (direct nvcc links do not consume host LDFLAGS)."
	@echo "SYCL settings: SYCL_CXX, SYCL_DPL_ROOT_PATH, SYCL_TARGET_FLAGS."
