
 # Poisson simple iterations sample code 

 ## Build
 Uses gcc for host targets and nvcc for CUDA target.
 To build all targets (including CUDA):
 ```
 make all
 ```
 To build separate target:
 ```
 make poisson_jacobi_<TARGET_NAME>.bin
 ```
 where ```<TARGET_NAME>``` is one of the following:
 * ```cpu``` for serial only host version
 * ```omp``` for openmp host version
 * ```cuda``` for CUDA version (requeres CUDA toolkit and nvcc available in path)

 ## Run

 To run 100x100x100 mesh with maximum 10000 iterations and target tolerance 1e-3 on serial version:
 ```
 time ./poisson_jacobi_cpu.bin 100 100 100 10000 1e-3
 ```

 To run with the same parameters on host with 6 openmp threads:
 ```
 time OMP_NUM_THREADS=6 ./poisson_jacobi_omp.bin 100 100 100 10000 1e-3
 ```

 To run with the same parameters on cuda device (chooses first in list, TODO add choise):
 ```
 time ./poisson_jacobi_cuda.bin 100 100 100 10000 1e-3
 ```

 ## Notes on realization

We need separate poisson_solver_impl.h file because of cuda. All cuda specific files (with kernels calls) must 
reside in the ```*.cu``` file. Otherwise nvcc will treat it as usual cpp code. At the same time gcc is unable to compile cu files (?). So we put implementation with kernels into separate poisson_solver_impl.h header and 
instantinate class in separate poisson_solver_cuda_inst.cu file for CUDA or poisson_solver_omp_inst.cpp for openmp (actually omp doesnot requere separate file here just for consistency). Serial version instantinated directly in poisson_jacobi.cpp that contains main function.