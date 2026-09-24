
if(PLATFORM STREQUAL "CUDA")
  add_executable(test_mat_cuda.bin test_mat.cu)
  target_link_libraries(test_mat_cuda.bin gtest_main)
endif()
add_executable(test_mat_host.bin test_mat_host.cpp)
target_link_libraries(test_mat_host.bin gtest_main)

if(PLATFORM STREQUAL "CUDA")
  scfd_add_test(test_mat_cuda static_mat CUDA)
endif()
scfd_add_test(test_mat_host static_mat SERIAL)
