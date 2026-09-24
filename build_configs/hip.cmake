# hipcc uses the active HIP installation/environment (AMD or NVIDIA).
# A machine-specific copy can supply an absolute compiler path and SDK flags.
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER hipcc CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(PLATFORM HIP CACHE STRING "Execution platform")
# MPI is automatically detected by run_ci.sh unless PLATFORM_MPI is specified.
set(SCFD_CI_REQUIRED OFF CACHE BOOL "Fail if this configuration is unavailable")
