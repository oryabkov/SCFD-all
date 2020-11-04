SCFD_INCLUDE=../../include
test_mat_cuda.bin: test_mat.cu
	nvcc -arch=sm_35 -I$(SCFD_INCLUDE) -std=c++11 test_mat.cu -o test_mat_cuda.bin
