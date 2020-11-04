SCFD_INCLUDE=../../include
test_vec.bin: test_vec.cpp
	g++ -I$(SCFD_INCLUDE) -std=c++11 test_vec.cpp -o test_vec.bin
test_vec_cuda.bin: test_vec.cu
	nvcc -I$(SCFD_INCLUDE) -std=c++11 test_vec.cu -o test_vec_cuda.bin

test_rect.bin: test_rect.cpp
	g++ -I$(SCFD_INCLUDE) -std=c++11 test_rect.cpp -o test_rect.bin