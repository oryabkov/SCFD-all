# Shared by the library tests and the small CI capability probe project.
# Select compilers before project(), using -D, -C, or a CMake toolchain file.
include_guard(GLOBAL)
include(CheckLanguage)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(scfd_check_sycl)
    cmake_push_check_state(RESET)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    if(SCFD_SYCL_TARGET)
        string(APPEND CMAKE_REQUIRED_FLAGS " -fsycl-targets=${SCFD_SYCL_TARGET}")
    endif()
    if(SCFD_SYCL_TARGET_BACKEND)
        string(APPEND CMAKE_REQUIRED_FLAGS
            " -Xsycl-target-backend \"${SCFD_SYCL_TARGET_BACKEND}\"")
    endif()
    set(CMAKE_REQUIRED_INCLUDES "${SCFD_ONEDPL_INCLUDE_DIR}")
    # Allow SDK paths/device flags to be corrected in an existing build tree.
    unset(SCFD_SYCL_COMPILER_WORKS CACHE)
    check_cxx_source_compiles("#include <sycl/sycl.hpp>
        #include <oneapi/dpl/execution>
        #include <oneapi/dpl/algorithm>
        int main() { sycl::queue queue; return 0; }" SCFD_SYCL_COMPILER_WORKS)
    cmake_pop_check_state()
    if(NOT SCFD_SYCL_COMPILER_WORKS)
        message(FATAL_ERROR
            "PLATFORM=SYCL requires a working SYCL C++ compiler and oneDPL headers. "
            "Select icpx as CMAKE_CXX_COMPILER on the first configure; "
            "set SCFD_ONEDPL_INCLUDE_DIR if needed.")
    endif()
endfunction()

# hipcc on NVIDIA invokes nvcc even for CXX sources. Discover a usable host
# OpenMP flag through that wrapper and forward it for both compiling and linking.
# This leaves runtime-library selection to the compiler, rather than assuming
# that every HIP toolchain uses GCC's libgomp.
function(scfd_hip_nvidia_openmp)
    cmake_push_check_state(RESET)
    if(SCFD_HIP_HOST_OPENMP_FLAGS)
        set(candidates "${SCFD_HIP_HOST_OPENMP_FLAGS}")
    else()
        set(candidates "-fopenmp" "-qopenmp" "-fiopenmp" "-fopenmp=libomp")
    endif()
    foreach(candidate IN LISTS candidates)
        separate_arguments(host_options NATIVE_COMMAND "${candidate}")
        set(forwarded_options "")
        foreach(option IN LISTS host_options)
            list(APPEND forwarded_options "-Xcompiler=${option}")
        endforeach()
        string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${forwarded_options}")
        set(CMAKE_REQUIRED_LINK_OPTIONS ${forwarded_options})
        # Check each candidate anew: check_cxx_source_compiles caches its result.
        unset(SCFD_HIP_OPENMP_WORKS CACHE)
        check_cxx_source_compiles("#include <omp.h>
            #ifndef _OPENMP
            #error OpenMP was not enabled
            #endif
            int main() { return omp_get_max_threads() < 1; }" SCFD_HIP_OPENMP_WORKS)
        if(SCFD_HIP_OPENMP_WORKS)
            set(OpenMP_CXX_FLAGS "${CMAKE_REQUIRED_FLAGS}" CACHE STRING
                "OpenMP flags forwarded through NVIDIA hipcc" FORCE)
            # The forwarded link flag makes the host compiler select its runtime.
            set(OpenMP_CXX_LIB_NAMES "" CACHE STRING
                "OpenMP runtime selected by NVIDIA hipcc's host compiler" FORCE)
            set(SCFD_HIP_OPENMP_LINK_OPTIONS "${forwarded_options}" PARENT_SCOPE)
            break()
        endif()
    endforeach()
    cmake_pop_check_state()
    if(NOT SCFD_HIP_OPENMP_WORKS)
        message(FATAL_ERROR
            "Cannot enable OpenMP through NVIDIA hipcc. "
            "Set SCFD_HIP_HOST_OPENMP_FLAGS to the host compiler's OpenMP flags.")
    endif()
endfunction()

macro(scfd_setup_toolchains)
    if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
        set(CMAKE_CXX_STANDARD 14)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(PLATFORM STREQUAL "CUDA")
        check_language(CUDA)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "PLATFORM=CUDA requires a CUDA compiler; put nvcc on PATH or set CMAKE_CUDA_COMPILER.")
        endif()
        enable_language(CUDA)
        if(NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDARD LESS 14)
            set(CMAKE_CUDA_STANDARD 14)
        endif()
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()

    if(PLATFORM STREQUAL "HIP")
        # NVIDIA hipcc identifies its host compiler as GNU, but nvcc accepts
        # -std=c++14, not CMake's default -std=gnu++14 spelling.
        set(CMAKE_CXX_EXTENSIONS OFF)
        cmake_push_check_state(RESET)
        foreach(_scfd_hip_platform NVIDIA AMD)
            unset(SCFD_HIP_${_scfd_hip_platform}_COMPILER_WORKS CACHE)
            check_cxx_source_compiles("#include <hip/hip_runtime.h>
                #ifndef __HIP_PLATFORM_${_scfd_hip_platform}__
                #error Wrong HIP platform
                #endif
                __global__ void scfd_hip_kernel() {}
                int main() { return 0; }" SCFD_HIP_${_scfd_hip_platform}_COMPILER_WORKS)
        endforeach()
        cmake_pop_check_state()
        if(SCFD_HIP_NVIDIA_COMPILER_WORKS)
            set(_scfd_detected_hip_platform nvidia)
        elseif(SCFD_HIP_AMD_COMPILER_WORKS)
            set(_scfd_detected_hip_platform amd)
        else()
            message(FATAL_ERROR
                "PLATFORM=HIP requires hipcc selected as CMAKE_CXX_COMPILER on the first configure.")
        endif()
        if(SCFD_HIP_PLATFORM AND NOT SCFD_HIP_PLATFORM STREQUAL _scfd_detected_hip_platform)
            message(FATAL_ERROR "SCFD_HIP_PLATFORM disagrees with the selected HIP compiler.")
        endif()
        set(SCFD_HIP_PLATFORM "${_scfd_detected_hip_platform}" CACHE STRING "HIP platform: amd or nvidia")
        if(SCFD_HIP_PLATFORM STREQUAL "nvidia")
            scfd_hip_nvidia_openmp()
        endif()
    endif()

    if(PLATFORM STREQUAL "SYCL")
        if(CMAKE_VERSION VERSION_LESS 3.20)
            message(FATAL_ERROR "SCFD's icpx/SYCL configuration requires CMake 3.20 or newer.")
        endif()
        set(SCFD_SYCL_TARGET "${SCFD_SYCL_TARGET}" CACHE STRING "SYCL device target triple(s)")
        set(SCFD_SYCL_TARGET_BACKEND "${SCFD_SYCL_TARGET_BACKEND}" CACHE STRING "SYCL device backend options")
        set(SCFD_ONEDPL_INCLUDE_DIR "${SCFD_ONEDPL_INCLUDE_DIR}" CACHE PATH "Optional oneDPL include directory")
        scfd_check_sycl()
    endif()

    # GPU examples contain OpenMP reference implementations. Pure SERIAL builds
    # have no OpenMP dependency.
    if(NOT PLATFORM STREQUAL "SERIAL")
        find_package(OpenMP REQUIRED COMPONENTS CXX)
    endif()
    if(PLATFORM STREQUAL "HIP" AND SCFD_HIP_PLATFORM STREQUAL "nvidia")
        set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
            INTERFACE_LINK_OPTIONS ${SCFD_HIP_OPENMP_LINK_OPTIONS})
    endif()
endmacro()

function(scfd_target_openmp target)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    if(PLATFORM STREQUAL "CUDA")
        separate_arguments(host_options NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
        foreach(option IN LISTS host_options)
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${option}>")
        endforeach()
    endif()
endfunction()

# Configure only the selected local backend. MPI flags/linking are intentionally
# left to the caller so that ordinary tests do not acquire an MPI dependency.
function(scfd_target_platform target)
    if(PLATFORM STREQUAL "SERIAL")
        target_compile_definitions(${target} PRIVATE PLATFORM_SERIAL_CPU)
    else()
        target_compile_definitions(${target} PRIVATE PLATFORM_${PLATFORM})
    endif()

    if(PLATFORM STREQUAL "OMP")
        scfd_target_openmp(${target})
    elseif(PLATFORM STREQUAL "CUDA")
        get_target_property(sources ${target} SOURCES)
        foreach(source IN LISTS sources)
            if(source MATCHES "\\.(cc|cpp|cxx|cu)$")
                set_source_files_properties("${source}" PROPERTIES LANGUAGE CUDA)
            endif()
        endforeach()
        # Before CMP0119, LANGUAGE alone does not tell nvcc that a .cpp file
        # contains CUDA kernels. Explicit mode also works with newer CMake.
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-x cu>")
        elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-x cuda>")
        endif()
    elseif(PLATFORM STREQUAL "SYCL")
        scfd_target_sycl(${target})
    endif()
endfunction()

function(scfd_target_sycl target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    set(options -fsycl)
    if(SCFD_SYCL_TARGET)
        list(APPEND options "-fsycl-targets=${SCFD_SYCL_TARGET}")
    endif()
    if(SCFD_SYCL_TARGET_BACKEND)
        list(APPEND options "SHELL:-Xsycl-target-backend \"${SCFD_SYCL_TARGET_BACKEND}\"")
    endif()
    target_compile_options(${target} PRIVATE ${options})
    target_link_options(${target} PRIVATE ${options})
    if(SCFD_ONEDPL_INCLUDE_DIR)
        target_include_directories(${target} PRIVATE "${SCFD_ONEDPL_INCLUDE_DIR}")
    endif()
endfunction()
