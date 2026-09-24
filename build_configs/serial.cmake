# Initial-cache defaults: cmake -C build_configs/serial.cmake ...
# Copy this file for a machine-specific compiler or MPI installation.
# Omit PLATFORM_MPI for automatic MPI detection by run_ci.sh; CMake itself
# defaults it to OFF. Set PLATFORM_MPI ON or OFF to request a fixed choice.
set(CMAKE_C_COMPILER gcc CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ compiler")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (keep assertions enabled)")
set(PLATFORM SERIAL CACHE STRING "Execution platform")
set(SCFD_CI_REQUIRED ON CACHE BOOL "Fail if this configuration is unavailable")
