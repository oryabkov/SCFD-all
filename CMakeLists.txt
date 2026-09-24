cmake_minimum_required(VERSION 3.14)
include(cmake/SCFDPlatform.cmake)
project(SCFD LANGUAGES C CXX)

include(cmake/SCFDToolchains.cmake)
scfd_setup_toolchains()

include_directories(include)
include(CTest)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
      googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
      )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    if(PLATFORM_MPI)
        find_package(MPI REQUIRED COMPONENTS CXX)
        if(NOT MPIEXEC_EXECUTABLE)
            message(FATAL_ERROR "PLATFORM_MPI=ON requires an MPI launcher (MPIEXEC_EXECUTABLE)")
        endif()
    endif()

    add_subdirectory(test)
endif()
