# Copyright (C) 2026 SCFD contributors

# This file is part of SCFD.

# SCFD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2 only of the License.

# SCFD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

SCFD_INCLUDE=../../include
.DEFAULT_GOAL := test_constant_data_cpp.bin
#TODO !WARNING this is temporal solution for stokes solver only!
NLOHMANN_JSON_INCLUDE=../../../json
MPI_INCLUDE_DIR=/usr/local/mpi/include
MPI_LIB_DIR=/usr/local/mpi/lib

CXX ?= g++
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++14
BUILD_DIR ?= build/scalar_traits
BOOST_ROOT ?=
SCALAR_CPPFLAGS = -I$(SCFD_INCLUDE)
ifneq ($(strip $(BOOST_ROOT)),)
SCALAR_CPPFLAGS += -I$(BOOST_ROOT)/include
endif
SCALAR_HEADERS = $(wildcard $(SCFD_INCLUDE)/scfd/utils/*scalar_traits.h)

.PHONY: test-scalar-traits test-boost-scalar-traits test-scalar-traits-cuda
$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/scalar_traits.bin: test_scalar_traits.cpp $(SCALAR_HEADERS) | $(BUILD_DIR)
	$(CXX) -std=c++17 $(CPPFLAGS) $(CXXFLAGS) $(SCALAR_CPPFLAGS) $< -o $@

$(BUILD_DIR)/boost_scalar_traits.bin: test_scalar_traits.cpp $(SCALAR_HEADERS) | $(BUILD_DIR)
	$(CXX) -std=c++17 $(CPPFLAGS) $(CXXFLAGS) $(SCALAR_CPPFLAGS) -DSCFD_TEST_WITH_BOOST $< -o $@

$(BUILD_DIR)/boost_scalar_traits_isolation.bin: test_scalar_traits.cpp $(SCALAR_HEADERS) | $(BUILD_DIR)
	$(CXX) -std=c++17 $(CPPFLAGS) $(CXXFLAGS) $(SCALAR_CPPFLAGS) -DSCFD_TEST_BOOST_ISOLATION $< -o $@

test-scalar-traits: $(BUILD_DIR)/scalar_traits.bin
	$<

test-boost-scalar-traits: $(BUILD_DIR)/boost_scalar_traits.bin $(BUILD_DIR)/boost_scalar_traits_isolation.bin
	$(BUILD_DIR)/boost_scalar_traits.bin
	$(BUILD_DIR)/boost_scalar_traits_isolation.bin

$(BUILD_DIR)/scalar_traits_cuda.bin: test_scalar_traits_cuda.cu $(SCALAR_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SCALAR_CPPFLAGS) $< -o $@

test-scalar-traits-cuda: $(BUILD_DIR)/scalar_traits_cuda.bin
	$<

test_constant_data_cpp.bin: test_constant_data.cpp 
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_constant_data.cpp -o test_constant_data_cpp.bin
test_constant_data_cu.bin: test_constant_data.cu test_constant_data2.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_constant_data.cu test_constant_data2.cu -o test_constant_data_cu.bin
logged_obj.bin: logged_obj.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) logged_obj.cpp -o logged_obj.bin
main_try_catch.bin: main_try_catch.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) main_try_catch.cpp -o main_try_catch.bin
cuda_test.bin: cuda_test.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) cuda_test.cu -o cuda_test.bin
cuda_test_presist.bin: cuda_test_presist.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) cuda_test_presist.cu -o cuda_test_presist.bin	
test_system_timer_event.bin: test_system_timer_event.cpp
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_system_timer_event.cpp -o test_system_timer_event.bin
test_log_basic_cformatted_wrap.bin: test_log_basic_cformatted_wrap.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_log_basic_cformatted_wrap.cpp -o test_log_basic_cformatted_wrap.bin
test_log_basic_cformatted_wrap_cuda.bin: test_log_basic_cformatted_wrap.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_log_basic_cformatted_wrap.cu -o test_log_basic_cformatted_wrap_cuda.bin
test_log_std_thread_safety.bin: test_log_std_thread_safety.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) -DSCFD_UTILS_LOG_GARANTEE_THREAD_SAFE=1 test_log_std_thread_safety.cpp -lpthread -o test_log_std_thread_safety.bin
test_log_std_thread_safety_cuda.bin: test_log_std_thread_safety.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) -DSCFD_UTILS_LOG_GARANTEE_THREAD_SAFE=1 test_log_std_thread_safety.cu -lpthread -o test_log_std_thread_safety_cuda.bin
test_log_mpi_cuda.bin: test_log_mpi_cuda.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) test_log_mpi_cuda.cu -lmpi -o test_log_mpi_cuda.bin
test_log_std_singleton.bin: test_log_std_singleton.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_log_std_singleton.cpp -o test_log_std_singleton.bin
test_format_json.bin: test_format_json.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) -I$(NLOHMANN_JSON_INCLUDE) test_format_json.cpp -o test_format_json.bin
test_todo:
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_todo.cpp -o test_todo.bin
	nvcc -std=c++14 -I$(SCFD_INCLUDE) test_todo.cu -o test_todo_cuda.bin
