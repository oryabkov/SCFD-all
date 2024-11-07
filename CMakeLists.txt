cmake_minimum_required(VERSION 3.14)
project(SCFD LANGUAGES C CXX)

include(CheckLanguage)

option(ENABLE_CUDA "Turns on CUDA tests build" OFF)
option(ENABLE_SYCL "Turns on SYCL tests build" OFF)

if(ENABLE_CUDA)
    message(STATUS "ENABLE_CUDA is turned ON - CUDA tests will be generated")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED 1)
  else()
    #TODO error instead?
    message(STATUS "WARNING No CUDA found! - CUDA tests whould be ommited")
    set(CUDA_ENABLED "")
  endif()
else()
    message(STATUS "ENABLE_CUDA is turned OFF - CUDA tests whould be ommited")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CUDA_ENABLED)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(ENABLE_SYCL)
    message(STATUS "ENABLE_SYCL is turned ON - SYCL tests will be generated")
    if(CMAKE_CXX_COMPILER MATCHES "icpx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    endif()
    set(SYCL_ENABLED 1)
else()
    message(STATUS "ENABLE_SYCL is turned OFF - SYCL tests whould be ommited")
endif()

find_package(OpenMP REQUIRED)

include_directories(include)

if(CUDA_ENABLED)
  #TODO use per target
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

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
