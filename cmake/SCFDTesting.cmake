include_guard(GLOBAL)

set(SCFD_TEST_TIMEOUT 120 CACHE STRING "Timeout in seconds for each SCFD test")
if(NOT SCFD_TEST_TIMEOUT MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "SCFD_TEST_TIMEOUT must be a positive integer")
endif()

function(scfd_test_properties name suite platform)
    string(TOLOWER "${platform}" platform_label)
    set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/cases/${name}")
    file(MAKE_DIRECTORY "${work_dir}")
    set_tests_properties(${name} PROPERTIES
        LABELS "${suite};${platform_label}"
        TIMEOUT "${SCFD_TEST_TIMEOUT}"
        WORKING_DIRECTORY "${work_dir}")
endfunction()

# Executable targets use the existing <test-name>.bin naming convention.
function(scfd_add_test name suite platform)
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}.bin> ${ARGN})
    scfd_test_properties(${name} "${suite}" "${platform}")
endfunction()

function(scfd_add_mpi_test name target suite platform ranks)
    add_test(NAME ${name}
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
                ${MPIEXEC_POSTFLAGS} ${ARGN})
    scfd_test_properties(${name} "${suite}" "${platform}")
    set_property(TEST ${name} APPEND PROPERTY LABELS mpi)
    set_tests_properties(${name} PROPERTIES PROCESSORS ${ranks})
endfunction()
