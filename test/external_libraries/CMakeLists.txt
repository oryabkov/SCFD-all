
if(PLATFORM STREQUAL "CUDA")
  add_executable(test_cusolver_wrap.bin test_cusolver_wrap.cu)
  target_link_libraries(test_cusolver_wrap.bin gtest cusolver cublas)

  scfd_add_test(test_cusolver_wrap external_libraries CUDA)
endif()

if(PLATFORM STREQUAL "HIP")
  find_package(hipblas REQUIRED CONFIG)
  add_executable(test_hipblas_wrap.bin test_hipblas_wrap.cpp)
  target_compile_features(test_hipblas_wrap.bin PRIVATE cxx_std_17)
  target_include_directories(test_hipblas_wrap.bin PRIVATE ${hipblas_INCLUDE_DIRS})
  # hipcc/NVIDIA rejects the versioned .so paths exported by roc::hipblas.
  target_link_directories(test_hipblas_wrap.bin PRIVATE $<TARGET_FILE_DIR:roc::hipblas>)
  target_link_libraries(test_hipblas_wrap.bin hipblas)
  scfd_add_test(test_hipblas_wrap external_libraries HIP)
endif()
