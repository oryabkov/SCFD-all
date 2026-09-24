cmake_minimum_required(VERSION 3.14)
project(SCFD LANGUAGES C CXX)

option(SCFD_WITH_CUDA   "Use CUDA"   OFF)
option(SCFD_WITH_SYCL   "Use SYCL"   OFF)
option(SCFD_WITH_HIP    "Use HIP"    OFF)
option(SCFD_WITH_MPI    "Enable MPI communication tests" OFF)
option(SCFD_WITH_TESTS  "Enable SCFD Tests Build" ON)

include(cmake/SCFDToolchains.cmake)
scfd_setup_toolchains()

include_directories(include)

if(SCFD_WITH_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
      )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

endif()

if(SCFD_WITH_TESTS)
    if(SCFD_WITH_MPI)
        find_package(MPI REQUIRED COMPONENTS CXX)
        if(NOT MPIEXEC_EXECUTABLE)
            message(FATAL_ERROR "SCFD_WITH_MPI requires an MPI launcher (MPIEXEC_EXECUTABLE)")
        endif()
    endif()

    enable_testing()
    add_subdirectory(test)
endif()
