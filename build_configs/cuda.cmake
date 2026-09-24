# Select a suitable architecture in a machine-specific copy if needed, e.g.
# set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
# CMake discovers nvcc; set CMAKE_CUDA_COMPILER here only to override discovery.
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(PLATFORM CUDA CACHE STRING "Execution platform")
# MPI is automatically detected by run_ci.sh unless PLATFORM_MPI is specified.
set(SCFD_CI_REQUIRED OFF CACHE BOOL "Fail if this configuration is unavailable")
