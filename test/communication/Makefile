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

CUDA_ROOT_PATH ?= /usr/local/cuda
MPI_ROOT_PATH ?= /usr/local/mpi
SYCL_DPL_ROOT_PATH ?= /opt/intel/oneapi-sycl-nvidia/dpl/2022.7

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
TRIVIAL_BUILD_DIR := build/trivial

platform_define_SERIAL := -DPLATFORM_SERIAL_CPU
platform_define_OMP := -DPLATFORM_OMP
platform_define_CUDA := -DPLATFORM_CUDA
platform_define_HIP := -DPLATFORM_HIP
platform_define_SYCL := -DPLATFORM_SYCL
COMMON_CPPFLAGS := -I$(SCFD_INCLUDE)
PLATFORM_CPPFLAGS := $(COMMON_CPPFLAGS) $(platform_define_$(PLATFORM))
ifeq ($(MPI),ON)
PLATFORM_CPPFLAGS += -DPLATFORM_MPI $(MPI_INCLUDE_FLAGS)
PLATFORM_LINK_FLAGS := $(MPI_LINK_FLAGS)
endif
PLATFORM_HEADERS := $(wildcard $(SCFD_INCLUDE)/scfd/platform/*.h) $(wildcard $(SCFD_INCLUDE)/scfd/backend/*.h)
COMMUNICATION_HEADERS := $(wildcard $(SCFD_INCLUDE)/scfd/communication/*.h)

TRIVIAL_BIN := $(TRIVIAL_BUILD_DIR)/test_trivial_comm.bin
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
PLATFORM_RUNTIME_BIN := $(BUILD_DIR)/test_platform_runtime.bin
PLATFORM_TRIVIAL_BIN := $(BUILD_DIR)/test_platform_trivial_comm.bin
PLATFORM_RECT_BIN := $(BUILD_DIR)/test_platform_rect_distributor.bin
PLATFORM_TENSOR_RECT_BIN := $(BUILD_DIR)/test_platform_rect_distributor_tensor.bin
PLATFORM_BINS := $(PLATFORM_RUNTIME_BIN)

ifeq ($(MPI),ON)
PLATFORM_BINS += $(PLATFORM_RECT_BIN) $(PLATFORM_TENSOR_RECT_BIN)
TEST_BINS := $(MPI_COMMON_BINS) $(PLATFORM_BINS)
else
ifeq ($(PLATFORM),SERIAL)
PLATFORM_BINS += $(PLATFORM_TRIVIAL_BIN)
endif
TEST_BINS := $(TRIVIAL_BIN) $(PLATFORM_BINS)
endif

MPI_LAUNCH = $(MPIRUN) $(MPI_TEST_RUN_FLAGS) $(if $(strip $(MPI_TEST_HOST)),--host $(MPI_TEST_HOST))

.PHONY: all run run-platform run-common clean help

all: $(TEST_BINS)

$(BUILD_DIR) $(MPI_COMMON_BUILD_DIR) $(TRIVIAL_BUILD_DIR):
	mkdir -p $@

$(TRIVIAL_BIN): test_trivial_comm.cpp $(COMMUNICATION_HEADERS) | $(TRIVIAL_BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(MPI_COMMON_BINS): $(COMMUNICATION_HEADERS)
$(MPI_COMMON_BUILD_DIR)/%.bin: %.cpp | $(MPI_COMMON_BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_CPPFLAGS) $< $(MPI_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@

$(PLATFORM_BINS): $(PLATFORM_HEADERS) $(COMMUNICATION_HEADERS)
$(PLATFORM_RUNTIME_BIN): ../backend/test_backend_runtime_common.h

ifeq ($(PLATFORM),CUDA)
ifeq ($(MPI),ON)
$(PLATFORM_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(PLATFORM_CPPFLAGS) -x cu -c $< -o $(@:.bin=.o)
	$(MPICXX) $(@:.bin=.o) $(MPI_LINK_FLAGS) $(LDFLAGS) $(CUDA_LINK_FLAGS) $(LDLIBS) -o $@
else
$(PLATFORM_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(PLATFORM_CPPFLAGS) -x cu $< $(CUDA_LINK_FLAGS) $(LDLIBS) -o $@
endif
else ifeq ($(PLATFORM),HIP)
$(PLATFORM_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) $(PLATFORM_CPPFLAGS) $< $(if $(filter ON,$(MPI)),$(HIP_MPI_LINK_FLAGS)) $(HIP_LDFLAGS) $(LDLIBS) -o $@
else ifeq ($(PLATFORM),SYCL)
$(PLATFORM_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(SYCL_CXX) $(CPPFLAGS) $(SYCLFLAGS) $(SYCL_TARGET_FLAGS) -fsycl $(PLATFORM_CPPFLAGS) $(SYCL_INCLUDE_FLAGS) $< $(PLATFORM_LINK_FLAGS) $(LDFLAGS) $(LDLIBS) -o $@
else
PLATFORM_CXX := $(if $(filter ON,$(MPI)),$(MPICXX),$(CXX))
$(PLATFORM_BINS): $(BUILD_DIR)/%.bin: %.cpp | $(BUILD_DIR)
	$(PLATFORM_CXX) $(CPPFLAGS) $(CXXFLAGS) $(PLATFORM_CPPFLAGS) $< $(PLATFORM_LINK_FLAGS) $(LDFLAGS) $(if $(filter OMP,$(PLATFORM)),$(OPENMP_FLAGS)) $(LDLIBS) -o $@
endif

run: run-common run-platform

ifeq ($(MPI),ON)
run-common: $(MPI_COMMON_BINS)
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
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_rect_distributor.bin $(MPI_RECT_ARGS)
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(MPI_COMMON_BUILD_DIR)/test_mpi_rect_distributor_tensor.bin $(MPI_RECT_ARGS)
run-platform: $(PLATFORM_BINS)
	$(MPI_LAUNCH) -n $(MPI_TEST_PROCS) $(PLATFORM_RUNTIME_BIN)
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(PLATFORM_RECT_BIN) $(MPI_RECT_ARGS)
	$(MPI_LAUNCH) -n $(MPI_RECT_PROCS) $(PLATFORM_TENSOR_RECT_BIN) $(MPI_RECT_ARGS)
else
run-common: $(TRIVIAL_BIN)
	$(TRIVIAL_BIN)
run-platform: $(PLATFORM_BINS)
	$(PLATFORM_RUNTIME_BIN)
ifeq ($(PLATFORM),SERIAL)
	$(PLATFORM_TRIVIAL_BIN)
endif
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
		'MPI=OFF runs the independent trivial communicator and local platform tests.' \
		'SERIAL MPI=OFF also runs the platform-selected trivial communicator test.' \
		'' \
		'Targets:' \
		'  all    Build both independent and platform-selected tests.' \
		'         Independent MPI tests go in build/mpi/, trivial in build/trivial/.' \
		'         Platform tests go in build/<platform>[-mpi]/.' \
		'  run    Build and run both test groups for the selected configuration.' \
		'  run-platform  Run platform runtime and relevant distributor tests.' \
		'  run-common    Run independent MPI tests or the non-MPI trivial test.' \
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
