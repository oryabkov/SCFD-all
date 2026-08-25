PLATFORM ?= SERIAL

ifneq ($(strip $(PLATFORM_MPI)),)
ifneq ($(origin MPI),undefined)
ifneq ($(MPI),$(PLATFORM_MPI))
$(error MPI=$(MPI) conflicts with PLATFORM_MPI=$(PLATFORM_MPI))
endif
else
MPI := $(PLATFORM_MPI)
endif
endif

MPI ?= OFF

VALID_PLATFORMS := SERIAL OMP CUDA HIP SYCL
VALID_MPI_VALUES := OFF ON

ifeq ($(filter $(PLATFORM),$(VALID_PLATFORMS)),)
$(error Unsupported PLATFORM='$(PLATFORM)'; choose one of: $(VALID_PLATFORMS))
endif
ifeq ($(filter $(MPI),$(VALID_MPI_VALUES)),)
$(error Unsupported MPI='$(MPI)'; choose one of: $(VALID_MPI_VALUES))
endif
ifneq ($(MPI),ON)
ifneq ($(PLATFORM),SERIAL)
$(error MPI=OFF supports only PLATFORM=SERIAL in test/communication)
endif
endif

CUDA_ROOT_PATH ?= /usr/local/cuda
MPI_ROOT_PATH ?= /usr/local/mpi
SYCL_DPL_ROOT_PATH ?= /opt/intel/oneapi-sycl-nvidia-2025.0.0/dpl/2022.7

SCFD_INCLUDE := ../../include
CXX ?= g++
MPICXX ?= $(MPI_ROOT_PATH)/bin/mpic++
MPIRUN ?= $(MPI_ROOT_PATH)/bin/mpiexec
NVCC ?= $(CUDA_ROOT_PATH)/bin/nvcc
HIPCC ?= hipcc
SYCL_CXX ?= $(if $(SYCL_PATH),$(SYCL_PATH)/bin/icpx,icpx)

CPPSTD ?= -std=c++14
CXXFLAGS ?= $(CPPSTD)
CUDAFLAGS ?= $(CPPSTD)
HIPFLAGS ?= $(CPPSTD)
SYCLFLAGS ?= -std=c++17
SYCL_TARGET_FLAGS ?=
SYCL_INCLUDE_FLAGS ?= -I$(SYCL_DPL_ROOT_PATH)/include
OPENMP_FLAGS ?= -fopenmp -lpthread
MPI_INCLUDE_FLAGS ?= -I$(MPI_ROOT_PATH)/include
MPI_LINK_FLAGS ?= -L$(MPI_ROOT_PATH)/lib -lmpi
HIP_MPI_LINK_FLAGS ?= $(MPI_LINK_FLAGS)
HIP_LDFLAGS ?=
CUDA_LINK_FLAGS ?= -ldl -L$(CUDA_ROOT_PATH)/lib64 -lcudart

MPI_TEST_PROCS ?= 4
MPI_TEST_HOST ?= localhost:$(MPI_TEST_PROCS)
MPI_TEST_RUN_FLAGS ?= --bind-to none
MPI_RECT_PROCS ?= 3
MPI_RECT_ARGS ?= 1 1 1 0 0
MPI_SPLIT_PROCS ?= 3 4
MPI_THREAD_LEVELS ?= 0 1 2 3

platform_dir_SERIAL := serial
platform_dir_OMP := omp
platform_dir_CUDA := cuda
platform_dir_HIP := hip
platform_dir_SYCL := sycl
CONFIG := $(platform_dir_$(PLATFORM))$(if $(filter ON,$(MPI)),-mpi)
BUILD_DIR := build/$(CONFIG)
MPI_COMMON_BUILD_DIR := build/mpi

platform_define_SERIAL := -DPLATFORM_SERIAL_CPU
platform_define_OMP := -DPLATFORM_OMP
platform_define_CUDA := -DPLATFORM_CUDA
platform_define_HIP := -DPLATFORM_HIP
platform_define_SYCL := -DPLATFORM_SYCL
BACKEND_CPPFLAGS := $(platform_define_$(PLATFORM)) -DSCFD_BACKEND_ENABLE_MPI
COMMON_CPPFLAGS := -I$(SCFD_INCLUDE)
MPI_CPPFLAGS := $(COMMON_CPPFLAGS) $(MPI_INCLUDE_FLAGS)

TRIVIAL_BIN := $(BUILD_DIR)/test_trivial_comm.bin
MPI_COMMON_NAMES := \
	test_mpi_rect_distributor \
	test_mpi_rect_distributor_tensor \
	test_sequential_mpi \
	test_mpi_binary_file \
	test_mpi_thread_multiple \
	test_mpi_comm_split \
	test_mpi_comm_collectives \
	test_mpi_comm_p2p \
	test_mpi_comm_calls
MPI_COMMON_BINS := $(addprefix $(MPI_COMMON_BUILD_DIR)/,$(addsuffix .bin,$(MPI_COMMON_NAMES)))
BACKEND_RUNTIME_BIN := $(BUILD_DIR)/test_backend_runtime_mpi.bin

ifeq ($(MPI),ON)
TEST_BINS := $(MPI_COMMON_BINS) $(BACKEND_RUNTIME_BIN)
else
TEST_BINS := $(TRIVIAL_BIN)
endif

MPI_LAUNCH = $(MPIRUN) $(MPI_TEST_RUN_FLAGS) $(if $(strip $(MPI_TEST_HOST)),--host $(MPI_TEST_HOST))

.PHONY: all run clean help

all: $(TEST_BINS)

$(BUILD_DIR) $(MPI_COMMON_BUILD_DIR):
	mkdir -p $@

$(TRIVIAL_BIN): test_trivial_comm.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(MPI_COMMON_BUILD_DIR)/%.bin: %.cpp | $(MPI_COMMON_BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@

ifeq ($(PLATFORM),CUDA)
BACKEND_RUNTIME_OBJ := $(BUILD_DIR)/test_backend_runtime_mpi.o

$(BACKEND_RUNTIME_OBJ): test_backend_runtime_mpi.cpp ../backend/test_backend_runtime_common.h | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(BACKEND_CPPFLAGS) $(MPI_CPPFLAGS) -x cu -c $< -o $@

$(BACKEND_RUNTIME_BIN): $(BACKEND_RUNTIME_OBJ)
	$(MPICXX) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(CUDA_LINK_FLAGS) $(LDLIBS) -o $@
else ifeq ($(PLATFORM),HIP)
$(BACKEND_RUNTIME_BIN): test_backend_runtime_mpi.cpp ../backend/test_backend_runtime_common.h | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) $(BACKEND_CPPFLAGS) $(MPI_CPPFLAGS) $< $(HIP_MPI_LINK_FLAGS) $(HIP_LDFLAGS) $(LDLIBS) -o $@
else ifeq ($(PLATFORM),SYCL)
$(BACKEND_RUNTIME_BIN): test_backend_runtime_mpi.cpp ../backend/test_backend_runtime_common.h | $(BUILD_DIR)
	$(SYCL_CXX) $(CPPFLAGS) $(SYCLFLAGS) $(SYCL_TARGET_FLAGS) -fsycl $(BACKEND_CPPFLAGS) $(MPI_CPPFLAGS) $(SYCL_INCLUDE_FLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@
else ifeq ($(PLATFORM),OMP)
$(BACKEND_RUNTIME_BIN): test_backend_runtime_mpi.cpp ../backend/test_backend_runtime_common.h | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(BACKEND_CPPFLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(OPENMP_FLAGS) $(LDLIBS) -o $@
else
$(BACKEND_RUNTIME_BIN): test_backend_runtime_mpi.cpp ../backend/test_backend_runtime_common.h | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(BACKEND_CPPFLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@
endif

ifeq ($(MPI),ON)
run: all
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_sequential_mpi.bin
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_binary_file.bin
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_thread_multiple.bin
	@set -e; for thread_level in $(MPI_THREAD_LEVELS); do \
		$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_thread_multiple.bin $$thread_level; \
	done
	@set -e; for split_procs in $(MPI_SPLIT_PROCS); do \
		$(MPI_LAUNCH) -n $$split_procs $(MPI_COMMON_BUILD_DIR)/test_mpi_comm_split.bin; \
	done
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_comm_collectives.bin
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_comm_p2p.bin
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_comm_calls.bin
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(BACKEND_RUNTIME_BIN)
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_rect_distributor.bin $(MPI_RECT_ARGS)
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_rect_distributor_tensor.bin $(MPI_RECT_ARGS)
else
run: all
	$(TRIVIAL_BIN)
endif

clean:
	$(RM) -r build *.bin *.o test_file_write.txt

help:
	@printf '%s\n' \
		'Build communication tests:' \
		'  make [PLATFORM=SERIAL|OMP|CUDA|HIP|SYCL] [MPI=OFF|ON]' \
		'' \
		'Defaults: PLATFORM=SERIAL MPI=OFF' \
		'PLATFORM_MPI=ON/OFF is accepted as an alias for MPI.' \
		'MPI=OFF is intentionally limited to the serial trivial communicator test.' \
		'' \
		'Targets:' \
		'  all    Build without running; shared MPI tests go in build/mpi/.' \
		'         The backend runtime goes in build/<platform>-mpi/.' \
		'  run    Build and run the selected test configuration.' \
		'  clean  Remove generated test binaries, objects, and output data.' \
		'  help   Show this message.' \
		'  Run make clean after changing compiler or low-level flag variables.' \
		'' \
		'MPI run settings:' \
		'  MPI_TEST_PROCS, MPI_TEST_HOST, MPI_TEST_RUN_FLAGS, MPI_RECT_PROCS,' \
		'  MPI_RECT_ARGS, MPI_SPLIT_PROCS, MPI_THREAD_LEVELS, MPIRUN.' \
		'  Distributor tests currently require MPI_RECT_PROCS=3.' \
		'' \
		'SYCL settings:' \
		'  SYCL_CXX, SYCL_DPL_ROOT_PATH, SYCL_TARGET_FLAGS.'
