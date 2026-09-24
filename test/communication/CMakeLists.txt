# Standalone communication tests do not depend on GoogleTest.
function(scfd_communication_target name source)
    add_executable(${name}.bin ${source})
    target_compile_features(${name}.bin PRIVATE cxx_std_14)
endfunction()

scfd_communication_target(test_trivial_comm test_trivial_comm.cpp)
scfd_add_test(test_trivial_comm communication SERIAL)

# This regression always verifies a non-MPI platform, even alongside MPI tests.
if(PLATFORM STREQUAL "SERIAL")
    scfd_communication_target(test_platform_trivial_comm_serial test_platform_trivial_comm.cpp)
    scfd_target_platform(test_platform_trivial_comm_serial.bin)
    scfd_add_test(test_platform_trivial_comm_serial communication SERIAL)
endif()

string(TOLOWER "${PLATFORM}" platform_suffix)
set(runtime test_platform_runtime_${platform_suffix})
scfd_communication_target(${runtime} test_platform_runtime.cpp)
scfd_target_platform(${runtime}.bin)
scfd_add_test(${runtime} communication "${PLATFORM}")

if(PLATFORM_MPI)
    # MPI definitions remain target-local: the local runtime above stays non-MPI.
    foreach(operation IN ITEMS runtime rect_distributor rect_distributor_tensor)
        set(target test_platform_${operation}_${platform_suffix}_mpi)
        scfd_communication_target(${target} test_platform_${operation}.cpp)
        scfd_target_platform(${target}.bin)
        target_compile_definitions(${target}.bin PRIVATE PLATFORM_MPI)
        target_link_libraries(${target}.bin PRIVATE MPI::MPI_CXX)

        if(operation STREQUAL "runtime")
            scfd_add_mpi_test(${target} ${target}.bin communication "${PLATFORM}" 4)
        else()
            scfd_add_mpi_test(${target} ${target}.bin communication "${PLATFORM}" 3 1 1 1 0 0)
        endif()
    endforeach()

    # These tests exercise the original host implementations without platform selection.
    foreach(target IN ITEMS test_sequential_mpi test_mpi_binary_file
                            test_mpi_thread_multiple test_mpi_comm_split
                            test_mpi_comm_collectives test_mpi_comm_p2p test_mpi_comm_calls
                            test_mpi_rect_distributor test_mpi_rect_distributor_tensor)
        scfd_communication_target(${target} ${target}.cpp)
        target_link_libraries(${target}.bin PRIVATE MPI::MPI_CXX)
    endforeach()

    foreach(target IN ITEMS test_sequential_mpi test_mpi_binary_file
                            test_mpi_comm_collectives test_mpi_comm_p2p test_mpi_comm_calls)
        scfd_add_mpi_test(${target} ${target}.bin communication SERIAL 4)
    endforeach()

    scfd_add_mpi_test(test_mpi_thread_multiple_default test_mpi_thread_multiple.bin communication SERIAL 4)
    foreach(level RANGE 0 3)
        scfd_add_mpi_test(test_mpi_thread_multiple_${level} test_mpi_thread_multiple.bin communication SERIAL 4 ${level})
    endforeach()
    foreach(ranks IN ITEMS 3 4)
        scfd_add_mpi_test(test_mpi_comm_split_${ranks} test_mpi_comm_split.bin communication SERIAL ${ranks})
    endforeach()
    foreach(target IN ITEMS test_mpi_rect_distributor test_mpi_rect_distributor_tensor)
        scfd_add_mpi_test(${target} ${target}.bin communication SERIAL 3 1 1 1 0 0)
    endforeach()
endif()
