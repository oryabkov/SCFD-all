cmake_minimum_required(VERSION 3.14)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/SCFDPlatform.cmake")
project(SCFDCapabilityProbe LANGUAGES C CXX)

option(SCFD_CI_PROBE_MPI "Probe MPI independently of the local backend" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# NVIDIA hipcc also compiles the independent MPI probe; its nvcc driver accepts
# -std=c++14 but not CMake's GNU-extension spelling -std=gnu++14.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
enable_testing()

# Bracket quoting preserves paths and launcher arguments without evaluating
# their contents when the main project loads this initial-cache file.
function(scfd_export_mpi_cache name type value)
    set(equals "=")
    string(FIND "${value}" "]${equals}]" delimiter_position)
    while(NOT delimiter_position EQUAL -1)
        string(APPEND equals "=")
        string(FIND "${value}" "]${equals}]" delimiter_position)
    endwhile()
    file(APPEND "${CMAKE_BINARY_DIR}/mpi.cmake"
        "set(${name} [${equals}[${value}]${equals}] CACHE ${type} \"Validated MPI configuration\" FORCE)\n")
endfunction()

if(SCFD_CI_PROBE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT MPIEXEC_EXECUTABLE)
        message(FATAL_ERROR "MPI capability detection requires an MPI launcher")
    endif()
    add_executable(mpi-capability.bin mpi.cpp)
    target_link_libraries(mpi-capability.bin PRIVATE MPI::MPI_CXX)

    add_test(NAME mpi_capability
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi-capability.bin>
                ${MPIEXEC_POSTFLAGS})
    set_tests_properties(mpi_capability PROPERTIES
        LABELS "capability;mpi"
        PROCESSORS 4)

    file(WRITE "${CMAKE_BINARY_DIR}/mpi.cmake" "# Validated MPI compiler/launcher pair and launcher flags.\n")
    scfd_export_mpi_cache(MPIEXEC_EXECUTABLE FILEPATH "${MPIEXEC_EXECUTABLE}")
    if(MPI_CXX_COMPILER)
        scfd_export_mpi_cache(MPI_CXX_COMPILER FILEPATH "${MPI_CXX_COMPILER}")
    else()
        # FindMPI can discover MPI built into the selected C++ compiler itself.
        scfd_export_mpi_cache(MPI_CXX_COMPILER FILEPATH "${CMAKE_CXX_COMPILER}")
    endif()
    foreach(name MPIEXEC_NUMPROC_FLAG MPIEXEC_PREFLAGS MPIEXEC_POSTFLAGS)
        scfd_export_mpi_cache(${name} STRING "${${name}}")
    endforeach()
else()
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/SCFDToolchains.cmake")
    scfd_setup_toolchains()
    if(PLATFORM STREQUAL "CUDA")
        add_executable(capability.bin cuda.cu)
    elseif(PLATFORM STREQUAL "HIP")
        add_executable(capability.bin hip.cpp)
    elseif(PLATFORM STREQUAL "SYCL")
        add_executable(capability.bin sycl.cpp)
    else()
        add_executable(capability.bin host.cpp)
    endif()
    scfd_target_platform(capability.bin)
    add_test(NAME platform_capability COMMAND capability.bin)
    set_tests_properties(platform_capability PROPERTIES LABELS "capability;${PLATFORM}")
endif()
