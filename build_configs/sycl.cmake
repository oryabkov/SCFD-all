# Activate the SYCL compiler environment before running CI.
# For an NVIDIA machine, a local copy can additionally set:
# set(SCFD_SYCL_TARGET nvptx64-nvidia-cuda CACHE STRING "SYCL targets")
# set(SCFD_SYCL_TARGET_BACKEND --cuda-gpu-arch=sm_75 CACHE STRING "Backend flags")
# set(SCFD_ONEDPL_INCLUDE_DIR /path/to/oneDPL/include CACHE PATH "oneDPL headers")
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER icpx CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(SCFD_WITH_CUDA OFF CACHE BOOL "Build CUDA tests")
set(SCFD_WITH_HIP OFF CACHE BOOL "Build HIP tests")
set(SCFD_WITH_SYCL ON CACHE BOOL "Build SYCL tests")
set(SCFD_CI_MPI OFF CACHE STRING "MPI capability policy: AUTO, ON, OFF")
set(SCFD_CI_REQUIRED OFF CACHE BOOL "Fail if this configuration is unavailable")
