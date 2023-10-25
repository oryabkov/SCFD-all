
add_executable(test_cusolver_wrap.bin test_cusolver_wrap.cu)
target_link_libraries(test_cusolver_wrap.bin gtest cusolver cublas)

add_test(NAME test_cusolver_wrap COMMAND test_cusolver_wrap.bin)
