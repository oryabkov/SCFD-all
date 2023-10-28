cmake_minimum_required(VERSION 3.14)
project(SCFD LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

#TODO use per target
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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
