
SCFD_INCLUDE_DIR = ../../include

HEADERS=poisson_solver.h current_dim.h current_poisson_solver.h
HEADERS_FUNC=$(HEADERS) poisson_solver_impl.h
HEADERS_LAMBDA=$(HEADERS) poisson_solver_lambda_impl.h
SOURSE=poisson_jacobi.cpp
INCLUDE_SCFD=-I$(SCFD_INCLUDE_DIR)
COMPILE_FLAGS=-O3 $(INCLUDE_SCFD)
COMPILE_FLAGS_LAMBDA=$(COMPILE_FLAGS) -DPOISSON_SOLVER_USE_LAMBDA

poisson_jacobi_func_cpu.bin: $(HEADERS) $(SOURSE)
	g++ $(COMPILE_FLAGS) $(INCLUDE_SCFD) -DPOISSON_SOLVER_SERIAL $(SOURSE) -o poisson_jacobi_func_cpu.bin
poisson_jacobi_func_omp.bin: $(HEADERS) $(SOURSE) poisson_solver_omp_inst.cpp
	g++ $(COMPILE_FLAGS) -fopenmp $(INCLUDE_SCFD) -DPOISSON_SOLVER_OMP $(SOURSE) poisson_solver_omp_inst.cpp -o poisson_jacobi_func_omp.bin
poisson_jacobi_func_cuda.bin: $(HEADERS) $(SOURSE) poisson_solver_cuda_inst.cu
	nvcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) -DPOISSON_SOLVER_CUDA $(SOURSE) poisson_solver_cuda_inst.cu -o poisson_jacobi_func_cuda.bin

poisson_jacobi_lambda_cpu.bin: $(HEADERS) $(SOURSE)
	g++ $(COMPILE_FLAGS_LAMBDA) $(INCLUDE_SCFD) -DPOISSON_SOLVER_SERIAL $(SOURSE) -o poisson_jacobi_lambda_cpu.bin
poisson_jacobi_lambda_omp.bin: $(HEADERS) $(SOURSE) poisson_solver_omp_inst.cpp
	g++ $(COMPILE_FLAGS_LAMBDA) -fopenmp $(INCLUDE_SCFD) -DPOISSON_SOLVER_OMP $(SOURSE) poisson_solver_omp_inst.cpp -o poisson_jacobi_lambda_omp.bin
poisson_jacobi_lambda_cuda.bin: $(HEADERS) $(SOURSE) poisson_solver_cuda_inst.cu
	nvcc $(COMPILE_FLAGS_LAMBDA) --extended-lambda $(INCLUDE_SCFD) -DPOISSON_SOLVER_CUDA $(SOURSE) poisson_solver_cuda_inst.cu -o poisson_jacobi_lambda_cuda.bin	

all: poisson_jacobi_func_cpu.bin poisson_jacobi_func_omp.bin poisson_jacobi_func_cuda.bin poisson_jacobi_lambda_cpu.bin poisson_jacobi_lambda_omp.bin poisson_jacobi_lambda_cuda.bin