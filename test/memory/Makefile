SCFD_INCLUDE=../../include
test_thrust_ptr_cuda.bin: test_thrust_ptr_cuda.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_thrust_ptr_cuda.cu -o test_thrust_ptr_cuda.bin

test_shared_locked_buffer.bin: test_shared_locked_buffer.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_shared_locked_buffer.cpp -o test_shared_locked_buffer_cpu.bin
