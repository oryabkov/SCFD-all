SCFD_INCLUDE=../../include

test_cusolver_wrap.bin: test_cusolver_wrap.cu
	nvcc -I${SCFD_INCLUDE} -O3 test_cusolver_wrap.cu -lcusolver -lcublas -o test_cusolver_wrap.bin