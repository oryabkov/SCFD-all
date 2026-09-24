# hipcc uses the active HIP installation/environment (AMD or NVIDIA).
# A machine-specific copy can supply an absolute compiler path and SDK flags.
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER hipcc CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(SCFD_WITH_CUDA OFF CACHE BOOL "Build CUDA tests")
set(SCFD_WITH_HIP ON CACHE BOOL "Build HIP tests")
set(SCFD_WITH_SYCL OFF CACHE BOOL "Build SYCL tests")
set(SCFD_CI_MPI OFF CACHE STRING "MPI capability policy: AUTO, ON, OFF")
set(SCFD_CI_REQUIRED OFF CACHE BOOL "Fail if this configuration is unavailable")
