# Include before project(): compiler defaults must be selected before CMake
# identifies the toolchain. Explicit compiler/environment/toolchain choices win.
include_guard(GLOBAL)

foreach(_scfd_legacy_option CUDA HIP SYCL MPI TESTS)
    if(DEFINED SCFD_WITH_${_scfd_legacy_option})
        message(FATAL_ERROR
            "SCFD_WITH_${_scfd_legacy_option} is no longer supported. "
            "Use PLATFORM=SERIAL|OMP|CUDA|HIP|SYCL, PLATFORM_MPI=ON|OFF, "
            "and BUILD_TESTING=ON|OFF in a fresh build directory.")
    endif()
endforeach()

set(PLATFORM SERIAL CACHE STRING "SCFD platform: SERIAL, OMP, CUDA, HIP or SYCL")
set_property(CACHE PLATFORM PROPERTY STRINGS SERIAL OMP CUDA HIP SYCL)
if(NOT PLATFORM MATCHES "^(SERIAL|OMP|CUDA|HIP|SYCL)$")
    message(FATAL_ERROR "Unsupported PLATFORM '${PLATFORM}'; choose SERIAL, OMP, CUDA, HIP or SYCL.")
endif()
option(PLATFORM_MPI "Enable MPI communication tests for the selected platform" OFF)
if(NOT PLATFORM_MPI MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "Unsupported PLATFORM_MPI '${PLATFORM_MPI}'; choose ON or OFF.")
endif()

# Different platforms can require different C++ compilers. Reuse of a configured
# tree would retain the old compiler and dependency checks.
if(DEFINED SCFD_CONFIGURED_PLATFORM AND NOT SCFD_CONFIGURED_PLATFORM STREQUAL PLATFORM)
    message(FATAL_ERROR
        "This build directory is configured for PLATFORM=${SCFD_CONFIGURED_PLATFORM}. "
        "Use a separate build directory for PLATFORM=${PLATFORM}.")
endif()
if(NOT CMAKE_SCRIPT_MODE_FILE)
    set(SCFD_CONFIGURED_PLATFORM "${PLATFORM}" CACHE INTERNAL "Platform of this build tree")
    if(NOT CMAKE_CXX_COMPILER AND NOT CMAKE_TOOLCHAIN_FILE
       AND "$ENV{CMAKE_TOOLCHAIN_FILE}" STREQUAL "" AND "$ENV{CXX}" STREQUAL "")
        if(PLATFORM STREQUAL "HIP")
            find_program(_scfd_default_cxx NAMES hipcc)
        elseif(PLATFORM STREQUAL "SYCL")
            find_program(_scfd_default_cxx NAMES icpx dpcpp)
        endif()
        if(PLATFORM MATCHES "^(HIP|SYCL)$")
            if(NOT _scfd_default_cxx)
                message(FATAL_ERROR
                    "No ${PLATFORM} C++ compiler found. Set CMAKE_CXX_COMPILER "
                    "or put hipcc (HIP) / icpx (SYCL) on PATH.")
            endif()
            set(CMAKE_CXX_COMPILER "${_scfd_default_cxx}" CACHE FILEPATH "C++ compiler")
            unset(_scfd_default_cxx CACHE)
        endif()
    endif()
endif()
