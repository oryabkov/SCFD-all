# Standalone communication tests do not depend on GoogleTest.
function(scfd_communication_target name source)
    add_executable(${name}.bin ${source})
    target_compile_features(${name}.bin PRIVATE cxx_std_14)
endfunction()

function(scfd_communication_mpi_test name target ranks)
    add_test(NAME ${name}
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}.bin>
                ${MPIEXEC_POSTFLAGS} ${ARGN})
    set_tests_properties(${name} PROPERTIES
        LABELS "communication;mpi"
        PROCESSORS ${ranks}
        TIMEOUT 60
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

scfd_communication_target(test_trivial_comm test_trivial_comm.cpp)
scfd_communication_target(test_platform_trivial_comm_serial test_platform_trivial_comm.cpp)
target_compile_definitions(test_platform_trivial_comm_serial.bin PRIVATE PLATFORM_SERIAL_CPU)

foreach(platform IN ITEMS serial omp)
    if(platform STREQUAL "serial")
        set(platform_definition PLATFORM_SERIAL_CPU)
    else()
        set(platform_definition PLATFORM_OMP)
    endif()

    set(runtime test_platform_runtime_${platform})
    scfd_communication_target(${runtime} test_platform_runtime.cpp)
    target_compile_definitions(${runtime}.bin PRIVATE ${platform_definition})
    if(platform STREQUAL "omp")
        target_link_libraries(${runtime}.bin PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(SCFD_WITH_MPI)
        foreach(operation IN ITEMS runtime rect_distributor rect_distributor_tensor)
            set(target test_platform_${operation}_${platform}_mpi)
            scfd_communication_target(${target} test_platform_${operation}.cpp)
            target_compile_definitions(${target}.bin PRIVATE ${platform_definition} PLATFORM_MPI)
            target_link_libraries(${target}.bin PRIVATE MPI::MPI_CXX)
            if(platform STREQUAL "omp")
                target_link_libraries(${target}.bin PRIVATE OpenMP::OpenMP_CXX)
            endif()

            if(operation STREQUAL "runtime")
                scfd_communication_mpi_test(${target} ${target} 4)
            else()
                scfd_communication_mpi_test(${target} ${target} 3 1 1 1 0 0)
            endif()
        endforeach()
    endif()
endforeach()

foreach(target IN ITEMS test_trivial_comm test_platform_trivial_comm_serial
                        test_platform_runtime_serial test_platform_runtime_omp)
    add_test(NAME ${target} COMMAND ${target}.bin)
    set_tests_properties(${target} PROPERTIES
        LABELS communication
        TIMEOUT 60
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

if(SCFD_WITH_MPI)
    foreach(target IN ITEMS test_sequential_mpi test_mpi_binary_file
                            test_mpi_thread_multiple test_mpi_comm_split
                            test_mpi_comm_collectives test_mpi_comm_p2p test_mpi_comm_calls
                            test_mpi_rect_distributor test_mpi_rect_distributor_tensor)
        scfd_communication_target(${target} ${target}.cpp)
        target_link_libraries(${target}.bin PRIVATE MPI::MPI_CXX)
    endforeach()

    foreach(target IN ITEMS test_sequential_mpi test_mpi_binary_file
                            test_mpi_comm_collectives test_mpi_comm_p2p test_mpi_comm_calls)
        scfd_communication_mpi_test(${target} ${target} 4)
    endforeach()

    scfd_communication_mpi_test(test_mpi_thread_multiple_default test_mpi_thread_multiple 4)
    foreach(level RANGE 0 3)
        scfd_communication_mpi_test(test_mpi_thread_multiple_${level} test_mpi_thread_multiple 4 ${level})
    endforeach()
    foreach(ranks IN ITEMS 3 4)
        scfd_communication_mpi_test(test_mpi_comm_split_${ranks} test_mpi_comm_split ${ranks})
    endforeach()
    foreach(target IN ITEMS test_mpi_rect_distributor test_mpi_rect_distributor_tensor)
        scfd_communication_mpi_test(${target} ${target} 3 1 1 1 0 0)
    endforeach()

    # This test writes a fixed filename; keep its output out of the source tree.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mpi_binary_file")
    set_tests_properties(test_mpi_binary_file PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mpi_binary_file")
endif()
