set(N_NUM_LIST 2 3 4 5)

if(PLATFORM STREQUAL "HIP")
  foreach(N_NUM ${N_NUM_LIST})

  add_executable(test_matmul_hip_${N_NUM}x${N_NUM}.bin test_matmul_hip.cpp)
  scfd_target_openmp(test_matmul_hip_${N_NUM}x${N_NUM}.bin)
  target_link_libraries(test_matmul_hip_${N_NUM}x${N_NUM}.bin PRIVATE gtest_main)
  target_compile_definitions(test_matmul_hip_${N_NUM}x${N_NUM}.bin PRIVATE N_NUM=${N_NUM} REAL=float)

  scfd_add_test(test_matmul_hip_${N_NUM}x${N_NUM} matmul HIP 1024 2 a)

  endforeach()
endif()

if(PLATFORM STREQUAL "CUDA")
  foreach(N_NUM ${N_NUM_LIST})

  add_executable(test_matmul_cuda_${N_NUM}x${N_NUM}.bin test_matmul_cuda.cu)
  scfd_target_openmp(test_matmul_cuda_${N_NUM}x${N_NUM}.bin)
  target_link_libraries(test_matmul_cuda_${N_NUM}x${N_NUM}.bin PRIVATE gtest_main)
  target_compile_definitions(test_matmul_cuda_${N_NUM}x${N_NUM}.bin PRIVATE N_NUM=${N_NUM} REAL=float)

  scfd_add_test(test_matmul_cuda_${N_NUM}x${N_NUM} matmul CUDA 1024 2 a)

  endforeach()

endif()
