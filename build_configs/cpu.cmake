# Initial-cache defaults: cmake -C build_configs/cpu.cmake ...
# Copy this file for a machine-specific compiler or MPI installation.
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(SCFD_WITH_CUDA OFF CACHE BOOL "Build CUDA tests")
set(SCFD_WITH_HIP OFF CACHE BOOL "Build HIP tests")
set(SCFD_WITH_SYCL OFF CACHE BOOL "Build SYCL tests")
set(SCFD_CI_MPI AUTO CACHE STRING "MPI capability policy: AUTO, ON, OFF")
set(SCFD_CI_REQUIRED ON CACHE BOOL "Fail if this configuration is unavailable")
