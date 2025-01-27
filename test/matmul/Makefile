LIST = float double
template:
	for i in $(LIST); do \
		for number in 2 3 4 5 ; do \
		    echo $$number ; \
			nvcc -arch=sm_80 -O3 test_matmul_cuda.cu  -Xcompiler -fopenmp -lpthread -I../../include -DN_NUM=$$number -DREAL=float -o a$$i$$number.bin ; \
	    done ; \
	done


#LIST = float double
#test:
#	for i in $(LIST); do \
#		echo $$i ; \
#	done
