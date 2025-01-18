if(HIP_ENABLED)
  add_executable(test_matmul_hip.bin test_matmul_hip.cpp)
  target_link_libraries(test_matmul_hip.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)
endif()
