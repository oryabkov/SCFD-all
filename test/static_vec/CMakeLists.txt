
add_executable(test_vec_cuda.bin test_vec_cuda.cu test_vec_cuda.cpp)
target_link_libraries(test_vec_cuda.bin gtest_main)
add_executable(test_vec_host.bin test_vec.cpp)
target_link_libraries(test_vec_host.bin gtest_main)
add_executable(test_rect_host.bin test_rect.cpp)
target_link_libraries(test_rect_host.bin gtest_main)

add_test(NAME test_vec_cuda COMMAND test_vec_cuda.bin)
add_test(NAME test_vec_host COMMAND test_vec_host.bin)
add_test(NAME test_rect_host COMMAND test_rect_host.bin)
