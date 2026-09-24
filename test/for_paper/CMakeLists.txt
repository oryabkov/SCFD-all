CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plot_resutls.py ${CMAKE_CURRENT_BINARY_DIR}/plot_resutls.py COPYONLY)

if(CUDA_ENABLED)
  add_executable(test_tensor_array_3d_cross_product.bin test_tensor_array_3d_cross_product.cu)
  scfd_target_openmp(test_tensor_array_3d_cross_product.bin)
  target_link_libraries(test_tensor_array_3d_cross_product.bin PRIVATE gtest_main)
endif()

add_executable(test_omp.bin test_omp.cpp)
scfd_target_openmp(test_omp.bin)
target_link_libraries(test_omp.bin PRIVATE gtest_main)

if(CUDA_ENABLED)
  add_executable(test_omp_cu.bin test_omp.cu)
  scfd_target_openmp(test_omp_cu.bin)
  target_link_libraries(test_omp_cu.bin PRIVATE gtest_main)
endif()

if(HIP_ENABLED)
  add_executable(test_hip.bin test_hip.cpp)
  scfd_target_openmp(test_hip.bin)
  target_link_libraries(test_hip.bin PRIVATE gtest_main)
endif()

if(SYCL_ENABLED)
  add_executable(test_sycl.bin test_sycl.cpp)
  scfd_target_sycl(test_sycl.bin)
  scfd_target_openmp(test_sycl.bin)
  target_link_libraries(test_sycl.bin PRIVATE gtest_main)
  target_compile_definitions(test_sycl.bin PUBLIC
      SCFD_ARRAYS_ORDINAL_TYPE=std::ptrdiff_t)
endif()

#TODO add calls to test_omp tests
if(CUDA_ENABLED)
  add_test(NAME test_tensor_array_3d_cross_product COMMAND test_tensor_array_3d_cross_product.bin 100000 100 a)
endif()

