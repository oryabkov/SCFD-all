
if(CUDA_ENABLED)
  add_executable(test_mat_cuda.bin test_mat.cu)
  target_link_libraries(test_mat_cuda.bin gtest_main)
endif()
add_executable(test_mat_host.bin test_mat_host.cpp)
target_link_libraries(test_mat_host.bin gtest_main)

if(CUDA_ENABLED)
  add_test(NAME test_mat_cuda COMMAND test_mat_cuda.bin)
endif()
add_test(NAME test_mat_host COMMAND test_mat_host.bin)
