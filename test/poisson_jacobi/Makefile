
SCFD_INCLUDE_DIR = ../../include

HEADERS=poisson_solver.h poisson_solver_impl.h current_dim.h current_poisson_solver.h
SOURSE=poisson_jacobi.cpp
COMPILE_FLAGS=-O3
INCLUDE_SCFD=-I$(SCFD_INCLUDE_DIR)

poisson_jacobi_cpu.bin: $(HEADERS) $(SOURSE)
	g++ $(COMPILE_FLAGS) $(INCLUDE_SCFD) -DPOISSON_SOLVER_SERIAL $(SOURSE) -o poisson_jacobi_cpu.bin
poisson_jacobi_omp.bin: $(HEADERS) $(SOURSE) poisson_solver_omp_inst.cpp
	g++ $(COMPILE_FLAGS) -fopenmp $(INCLUDE_SCFD) -DPOISSON_SOLVER_OMP $(SOURSE) poisson_solver_omp_inst.cpp -o poisson_jacobi_omp.bin
poisson_jacobi_cuda.bin: $(HEADERS) $(SOURSE) poisson_solver_cuda_inst.cu
	nvcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) -DPOISSON_SOLVER_CUDA $(SOURSE) poisson_solver_cuda_inst.cu -o poisson_jacobi_cuda.bin