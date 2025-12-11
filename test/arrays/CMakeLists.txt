if(CUDA_ENABLED)
  add_executable(test_cuda_unified_arrays.bin test_cuda_unified_arrays.cu)
  target_compile_options(test_cuda_unified_arrays.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
  target_link_libraries(test_cuda_unified_arrays.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)

  add_executable(test_tensor_array_nd_cuda.bin test_tensor_array_nd.cu)
  target_link_libraries(test_tensor_array_nd_cuda.bin gtest_main)

  add_executable(test_tensor_array_nd_visible_cuda.bin test_tensor_array_nd_visible.cu)
  target_link_libraries(test_tensor_array_nd_visible_cuda.bin gtest_main)
  target_compile_features(test_tensor_array_nd_visible_cuda.bin PRIVATE cxx_std_14)
  target_compile_options(test_tensor_array_nd_visible_cuda.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp> --expt-relaxed-constexpr)

  add_executable(test_custom_index_arranger_cuda.bin test_custom_index_arranger.cu)
  target_link_libraries(test_custom_index_arranger_cuda.bin gtest_main)
  target_compile_features(test_custom_index_arranger_cuda.bin PRIVATE cxx_std_14)
  target_compile_options(test_custom_index_arranger_cuda.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp> --expt-relaxed-constexpr)  
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


if(CUDA_ENABLED)
  add_test(NAME test_cuda_unified_arrays COMMAND test_cuda_unified_arrays.bin)
  add_test(NAME test_tensor_array_nd_cuda COMMAND test_tensor_array_nd_cuda.bin 10000 100 10)
  add_test(NAME test_tensor_array_nd_visible_cuda COMMAND test_tensor_array_nd_visible_cuda.bin)
  add_test(NAME test_custom_index_arranger_cuda COMMAND test_custom_index_arranger_cuda.bin)
endif()

add_test(NAME test_tensor_host COMMAND test_tensor_host.bin)
add_test(NAME test_tensor_array_nd_host COMMAND test_tensor_array_nd_host.bin 100000)
add_test(NAME test_tensor_array_nd_syntax_host COMMAND test_tensor_array_nd_syntax_host.bin)
add_test(NAME test_last_index_fast_arranger_host COMMAND test_last_index_fast_arranger_host.bin)
add_test(NAME test_custom_index_arranger_host COMMAND test_custom_index_arranger_host.bin)

add_subdirectory(detail)
