set(N_NUM_LIST 2 3 4 5)

if(HIP_ENABLED)
  foreach(N_NUM ${N_NUM_LIST})

  add_executable(test_matmul_hip_${N_NUM}x${N_NUM}.bin test_matmul_hip.cpp)
  target_link_libraries(test_matmul_hip_${N_NUM}x${N_NUM}.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)
  target_compile_definitions(test_matmul_hip_${N_NUM}x${N_NUM}.bin PRIVATE N_NUM=${N_NUM})

  endforeach()
endif()

if(CUDA_ENABLED)
  foreach(N_NUM ${N_NUM_LIST})

  add_executable(test_matmul_cuda_${N_NUM}x${N_NUM}.bin test_matmul_cuda.cu)
  target_link_libraries(test_matmul_cuda_${N_NUM}x${N_NUM}.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)
  target_compile_definitions(test_matmul_cuda_${N_NUM}x${N_NUM}.bin PRIVATE N_NUM=${N_NUM})

  endforeach()
endif()
