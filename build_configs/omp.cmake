# Initial-cache defaults for the OpenMP platform.
# MPI is automatically detected by run_ci.sh unless PLATFORM_MPI is specified.
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(PLATFORM OMP CACHE STRING "Execution platform")
set(SCFD_CI_REQUIRED ON CACHE BOOL "Fail if this configuration is unavailable")
