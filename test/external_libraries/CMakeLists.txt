
if(PLATFORM STREQUAL "CUDA")
  add_executable(test_cusolver_wrap.bin test_cusolver_wrap.cu)
  target_link_libraries(test_cusolver_wrap.bin gtest cusolver cublas)

  scfd_add_test(test_cusolver_wrap external_libraries CUDA)
endif()
