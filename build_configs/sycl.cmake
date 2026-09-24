# Activate the SYCL compiler environment before running CI.
# For an NVIDIA machine, a local copy can additionally set:
# set(SCFD_SYCL_TARGET nvptx64-nvidia-cuda CACHE STRING "SYCL targets")
# set(SCFD_SYCL_TARGET_BACKEND --cuda-gpu-arch=sm_75 CACHE STRING "Backend flags")
# set(SCFD_ONEDPL_INCLUDE_DIR /path/to/oneDPL/include CACHE PATH "oneDPL headers")
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER icpx CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(PLATFORM SYCL CACHE STRING "Execution platform")
# MPI is automatically detected by run_ci.sh unless PLATFORM_MPI is specified.
set(SCFD_CI_REQUIRED OFF CACHE BOOL "Fail if this configuration is unavailable")
