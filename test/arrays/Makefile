SCFD_INCLUDE=../../include
test_last_index_fast_arranger.bin: test_last_index_fast_arranger.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_last_index_fast_arranger.cpp -o test_last_index_fast_arranger.bin
test_tensor.bin: test_tensor.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_tensor.cpp -o test_tensor.bin
test_tensor_array_nd_syntax.bin: test_tensor_array_nd_syntax.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_tensor_array_nd_syntax.cpp -o test_tensor_array_nd_syntax.bin
test_tensor_array_nd_cuda.bin: test_tensor_array_nd.cu
	nvcc -O3 -I$(SCFD_INCLUDE) -std=c++11 test_tensor_array_nd.cu -o test_tensor_array_nd_cuda.bin
test_tensor_array_nd_cuda_deb.bin: test_tensor_array_nd.cu
	nvcc -g -I$(SCFD_INCLUDE) -std=c++11 test_tensor_array_nd.cu -o test_tensor_array_nd_cuda_deb.bin
test_tensor_array_nd_cpp.bin: test_tensor_array_nd.cpp
	g++ -O3 -I$(SCFD_INCLUDE) --std c++11 test_tensor_array_nd.cpp -o test_tensor_array_nd_cpp.bin
test_tensor_array_nd_cpp_deb.bin: test_tensor_array_nd.cpp
	g++ -g -I$(SCFD_INCLUDE) --std c++11 test_tensor_array_nd.cpp -o test_tensor_array_nd_cpp_deb.bin
test_tensor_array_nd_visible_cuda.bin: test_tensor_array_nd_visible.cu
	nvcc -O3 -I$(SCFD_INCLUDE) -std=c++11 test_tensor_array_nd_visible.cu -o test_tensor_array_nd_visible_cuda.bin

all: test_last_index_fast_arranger.bin test_tensor.bin test_tensor_array_nd_syntax.bin test_tensor_array_nd_cuda.bin test_tensor_array_nd_cpp.bin test_tensor_array_nd_visible_cuda.bin