CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/plot_resutls.py ${CMAKE_CURRENT_BINARY_DIR}/plot_resutls.py COPYONLY)

add_executable(test_tensor_array_3d_cross_product.bin test_tensor_array_3d_cross_product.cu)
target_compile_options(test_tensor_array_3d_cross_product.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(test_tensor_array_3d_cross_product.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)

add_executable(test_omp.bin test_omp.cpp)
target_link_libraries(test_omp.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)

add_executable(test_omp_cu.bin test_omp.cu)
target_compile_options(test_omp_cu.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(test_omp_cu.bin PRIVATE OpenMP::OpenMP_CXX gtest_main)


add_test(NAME test_tensor_array_3d_cross_product COMMAND test_tensor_array_3d_cross_product.bin 100000 100 a)


