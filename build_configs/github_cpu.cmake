# GitHub installs OpenMPI before invoking the shared CI runner.
# Set this before cpu.cmake so its AUTO default cannot replace it.
set(SCFD_CI_MPI ON CACHE STRING "MPI capability policy: AUTO, ON, OFF")
include("${CMAKE_CURRENT_LIST_DIR}/cpu.cmake")
