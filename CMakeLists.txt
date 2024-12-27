cmake_minimum_required(VERSION 3.14)
project(SCFD LANGUAGES C)

include(CheckLanguage)

# Set default compile optimization flag
set(SCFD_BUILD_TYPE "Release" CACHE STRING
    "Optimization flags: set to Debug, Release, RelWithDebInfo, or MinSizeRel")

option(SCFD_WITH_CUDA   "Use CUDA"   OFF)
option(SCFD_WITH_SYCL   "Use SYCL"   OFF)
option(SCFD_WITH_OPENMP "Use OpenMP" OFF)
option(SCFD_WITH_HIP    "Use HIP"    OFF) #SOON
option(SCFD_WITH_TESTS  "Enable SCFD Tests Build" OFF)

include_directories(include)

if(SCFD_WITH_CUDA)
    enable_language(CXX)
    message(STATUS "Enabled support for CXX")

    if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 11)
        set(CMAKE_CXX_STANDARD 11)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")    
    
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "SCFD with CUDA support")
        
        set(CUDA_ENABLED 1)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(WARNING "No CUDA compiler found!")
    endif()
endif()

if(SCFD_WITH_SYCL)
    enable_language(CXX)
    message(STATUS "Enabled support for CXX")

    # Enforce C++17
    if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
        set(CMAKE_CXX_STANDARD 17)
    endif ()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")

    find_program(INTEL_DPCPP_COMPILER icpx)
    if(INTEL_DPCPP_COMPILER)
        set(CMAKE_CXX_COMPILER ${INTEL_DPCPP_COMPILER})
        message(STATUS "SCFD with SYCL support")
        message(STATUS "Found intel dpcpp compiler at: ${INTEL_DPCPP_COMPILER}")

        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsycl>")
        add_link_options("-fsycl")

        set(SYCL_ENABLED 1)
        if(SCFD_SYCL_TARGET)
            add_link_options("SHELL:-fsycl-targets=${SCFD_SYCL_TARGET}")
        endif()
        if(SCFD_SYCL_TARGET_BACKEND)
            add_link_options("SHELL:-Xsycl-target-backend '${SCFD_SYCL_TARGET_BACKEND}'")
        endif()
    else()
        message(WARNING "No SYCL compiler found!")
    endif()
endif()

if(SCFD_WITH_OPENMP)
    enable_language(CXX)
    find_package(OpenMP REQUIRED)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
    add_link_options("-fopenmp")
endif()

if(SCFD_WITH_TESTS)
    # Following is direct copy from 
    # https://github.com/google/googletest/blob/master/googletest/README.md
    # Incorporating Into An Existing CMake Project section
    # With url cahnged from commit hash for release-1.12.1 - because it's latest
    # version that doesnot require C++14 and example target been commented out
    # BEGIN GTEST SECTION
    
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
      )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    ## Now simply link against gtest or gtest_main as needed. Eg
    #add_executable(example example.cpp)
    #target_link_libraries(example gtest_main)
    #add_test(NAME example_test COMMAND example)

    # END GTEST SECTION

    enable_testing()
    add_subdirectory(test)
endif()
