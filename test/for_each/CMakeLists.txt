

#set (SOURCEFILES test_for_each_nd.cu)
#set (HEADERFILES for_each_ndim_openmp_impl.h for_each_enums.h for_each_ndim_cuda_impl.cuh for_each_storage_types.h for_each_ndim.h)

#ordinal CUDA/host memory tests

#TODO !! rename cu to cpp and enable it for host case
if(CUDA_ENABLED)
add_executable( test_for_each_host.bin test_for_each.cu )
add_executable( test_for_each_nd_host.bin test_for_each_nd.cu )
target_compile_definitions(test_for_each_host.bin PUBLIC TEST_HOST)
target_compile_definitions(test_for_each_nd_host.bin PUBLIC TEST_HOST)
endif()

if(CUDA_ENABLED)
  add_executable( test_for_each_cuda.bin test_for_each.cu )
  add_executable( test_for_each_nd_cuda.bin test_for_each_nd.cu )
  target_compile_definitions(test_for_each_cuda.bin PUBLIC TEST_CUDA)
  target_compile_definitions(test_for_each_nd_cuda.bin PUBLIC TEST_CUDA)
endif()

#TODO !! rename cu to cpp and enable it for host case
if(CUDA_ENABLED)
add_executable( test_for_each_omp.bin test_for_each.cu )
add_executable( test_for_each_nd_omp.bin test_for_each_nd.cu )
target_compile_definitions(test_for_each_omp.bin PUBLIC TEST_OPENMP)
target_compile_definitions(test_for_each_nd_omp.bin PUBLIC TEST_OPENMP)
target_link_libraries( test_for_each_omp.bin PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries( test_for_each_nd_omp.bin PUBLIC OpenMP::OpenMP_CXX)
endif()

#TODO !! rename cu to cpp and enable it for host case
if(CUDA_ENABLED)
add_test( NAME test_for_each_host.bin COMMAND ./test_for_each_host.bin)
add_test( NAME test_for_each_nd_host.bin COMMAND ./test_for_each_nd_host.bin)
endif()
if(CUDA_ENABLED)
  add_test( NAME test_for_each_cuda.bin COMMAND ./test_for_each_cuda.bin)
  add_test( NAME test_for_each_nd_cuda.bin COMMAND ./test_for_each_nd_cuda.bin)
endif()
#TODO !! rename cu to cpp and enable it for host case
if(CUDA_ENABLED)
add_test( NAME test_for_each_omp.bin COMMAND ./test_for_each_omp.bin)
add_test( NAME test_for_each_nd_omp.bin COMMAND ./test_for_each_nd_omp.bin)
endif()

#CUDA unified memory tests

if(CUDA_ENABLED)
  add_executable( test_for_each_unified_host.bin test_for_each.cu )
  add_executable( test_for_each_nd_unified_host.bin test_for_each_nd.cu )
  target_compile_definitions(test_for_each_unified_host.bin PUBLIC TEST_UNIFIED_HOST)
  target_compile_definitions(test_for_each_nd_unified_host.bin PUBLIC TEST_UNIFIED_HOST)

  add_executable( test_for_each_unified_cuda.bin test_for_each.cu )
  add_executable( test_for_each_nd_unified_cuda.bin test_for_each_nd.cu )
  target_compile_definitions(test_for_each_unified_cuda.bin PUBLIC TEST_UNIFIED_CUDA)
  target_compile_definitions(test_for_each_nd_unified_cuda.bin PUBLIC TEST_UNIFIED_CUDA)

  add_executable( test_for_each_unified_omp.bin test_for_each.cu )
  add_executable( test_for_each_nd_unified_omp.bin test_for_each_nd.cu )
  target_compile_definitions(test_for_each_unified_omp.bin PUBLIC TEST_UNIFIED_OPENMP)
  target_compile_definitions(test_for_each_nd_unified_omp.bin PUBLIC TEST_UNIFIED_OPENMP)
  target_link_libraries( test_for_each_unified_omp.bin PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries( test_for_each_nd_unified_omp.bin PUBLIC OpenMP::OpenMP_CXX)

  add_test( NAME test_for_each_unified_host.bin COMMAND ./test_for_each_unified_host.bin)
  add_test( NAME test_for_each_nd_unified_host.bin COMMAND ./test_for_each_nd_unified_host.bin)
  add_test( NAME test_for_each_unified_cuda.bin COMMAND ./test_for_each_unified_cuda.bin)
  add_test( NAME test_for_each_nd_unified_cuda.bin COMMAND ./test_for_each_nd_unified_cuda.bin)
  add_test( NAME test_for_each_unified_omp.bin COMMAND ./test_for_each_unified_omp.bin)
  add_test( NAME test_for_each_nd_unified_omp.bin COMMAND ./test_for_each_nd_unified_omp.bin)
endif()

#SYCL test
if(SYCL_ENABLED)
    add_executable( test_for_each_sycl.bin           test_for_each_sycl.cpp )
    add_test( NAME  test_for_each_sycl.bin COMMAND ./test_for_each_sycl.bin )
    add_executable( test_for_each_nd_sycl.bin           test_for_each_nd_sycl.cpp )
    add_test( NAME  test_for_each_nd_sycl.bin COMMAND ./test_for_each_nd_sycl.bin )
endif()

#HIP test
if(HIP_ENABLED)
    add_executable( test_for_each_hip.bin           test_for_each_hip.cpp )
    add_test( NAME  test_for_each_hip.bin COMMAND ./test_for_each_hip.bin )
endif()

add_executable( test_for_each_func_macro.bin test_for_each_func_macro.cpp )
add_test( NAME test_for_each_func_macro.bin COMMAND ./test_for_each_func_macro.bin)
