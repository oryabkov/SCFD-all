

#set (SOURCEFILES test_for_each_nd.cu)
#set (HEADERFILES for_each_ndim_openmp_impl.h for_each_enums.h for_each_ndim_cuda_impl.cuh for_each_storage_types.h for_each_ndim.h)

add_executable( test_for_each.bin test_for_each.cu )
add_executable( test_for_each_nd.bin test_for_each_nd.cu )
add_executable( test_for_each_func_macro.bin test_for_each_func_macro.cpp )

target_link_libraries( test_for_each.bin PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries( test_for_each_nd.bin PUBLIC OpenMP::OpenMP_CXX)

add_test( NAME test_for_each.bin COMMAND ./test_for_each.bin)
add_test( NAME test_for_each_nd.bin COMMAND ./test_for_each_nd.bin)
add_test( NAME test_for_each_func_macro.bin COMMAND ./test_for_each_func_macro.bin)
