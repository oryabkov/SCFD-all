SCFD_INCLUDE=../../include

test_cusolver_wrap.bin: test_cusolver_wrap.cu
	nvcc -I${SCFD_INCLUDE} -O3 test_cusolver_wrap.cu -lcusolver -lcublas -o test_cusolver_wrap.bin

test_fail_cusolver_wrap_glob_init.bin: test_fail_cusolver_wrap_glob_init.cu
	nvcc -I${SCFD_INCLUDE} -O3 test_fail_cusolver_wrap_glob_init.cu -lcusolver -lcublas -o test_fail_cusolver_wrap_glob_init.bin

test_gesv_with_defect_matrix.bin: test_gesv_with_defect_matrix.cu	
	nvcc -arch=sm_75 -Xcompiler -fopenmp -lpthread -I${SCFD_INCLUDE} -O3 test_gesv_with_defect_matrix.cu -lcusolver -lcublas -o test_gesv_with_defect_matrix.bin
	