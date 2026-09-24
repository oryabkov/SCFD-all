if(PLATFORM STREQUAL "CUDA")
  add_executable(test_cuda_unified_arrays.bin test_cuda_unified_arrays.cu)
  scfd_target_openmp(test_cuda_unified_arrays.bin)
  target_link_libraries(test_cuda_unified_arrays.bin PRIVATE gtest_main)

  add_executable(test_tensor_array_nd_cuda.bin test_tensor_array_nd.cu)
  target_link_libraries(test_tensor_array_nd_cuda.bin gtest_main)

  add_executable(test_tensor_array_nd_visible_cuda.bin test_tensor_array_nd_visible.cu)
  target_link_libraries(test_tensor_array_nd_visible_cuda.bin gtest_main)
  
  add_executable(test_custom_index_arranger_cuda.bin test_custom_index_arranger.cu)
  target_link_libraries(test_custom_index_arranger_cuda.bin gtest_main)
endif()

add_compile_options(-I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(test_tensor_host.bin test_tensor.cpp)
target_link_libraries(test_tensor_host.bin gtest_main)

add_executable(test_tensor_array_nd_host.bin test_tensor_array_nd.cpp)
target_link_libraries(test_tensor_array_nd_host.bin gtest_main)

add_executable(test_tensor_array_nd_syntax_host.bin test_tensor_array_nd_syntax.cpp)
target_link_libraries(test_tensor_array_nd_syntax_host.bin gtest_main)

add_executable(test_last_index_fast_arranger_host.bin test_last_index_fast_arranger.cpp)
target_link_libraries(test_last_index_fast_arranger_host.bin gtest_main)

add_executable(test_custom_index_arranger_host.bin test_custom_index_arranger.cpp)
target_link_libraries(test_custom_index_arranger_host.bin gtest_main)
target_compile_features(test_custom_index_arranger_host.bin PRIVATE cxx_std_14)


if(PLATFORM STREQUAL "CUDA")
  scfd_add_test(test_cuda_unified_arrays arrays CUDA)
  # Arguments are the two dimensions and repetitions; bound the allocation for CI.
  scfd_add_test(test_tensor_array_nd_cuda arrays CUDA 1024 32 2)
  scfd_add_test(test_tensor_array_nd_visible_cuda arrays CUDA)
  scfd_add_test(test_custom_index_arranger_cuda arrays CUDA)
endif()

scfd_add_test(test_tensor_host arrays SERIAL)
scfd_add_test(test_tensor_array_nd_host arrays SERIAL 100000)
scfd_add_test(test_tensor_array_nd_syntax_host arrays SERIAL)
scfd_add_test(test_last_index_fast_arranger_host arrays SERIAL)
scfd_add_test(test_custom_index_arranger_host arrays SERIAL)

add_subdirectory(detail)
