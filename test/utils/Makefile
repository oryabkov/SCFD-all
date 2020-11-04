SCFD_INCLUDE=../../include
test_constant_data_cpp.bin: test_constant_data.cpp 
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_constant_data.cpp -o test_constant_data_cpp.bin
test_constant_data_cu.bin: test_constant_data.cu test_constant_data2.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_constant_data.cu test_constant_data2.cu -o test_constant_data_cu.bin
logged_obj.bin: logged_obj.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) logged_obj.cpp -o logged_obj.bin
main_try_catch.bin: main_try_catch.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) main_try_catch.cpp -o main_try_catch.bin
cuda_test.bin: cuda_test.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) cuda_test.cu -o cuda_test.bin
test_system_timer_event.bin: test_system_timer_event.cpp
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_system_timer_event.cpp -o test_system_timer_event.bin
test_log_basic_cformatted_wrap.bin: test_log_basic_cformatted_wrap.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) test_log_basic_cformatted_wrap.cpp -o test_log_basic_cformatted_wrap.bin
test_log_basic_cformatted_wrap_cuda.bin: test_log_basic_cformatted_wrap.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) test_log_basic_cformatted_wrap.cu -o test_log_basic_cformatted_wrap_cuda.bin
test_log_std_thread_safety.bin: test_log_std_thread_safety.cpp
	g++ -std=c++11 -I$(SCFD_INCLUDE) -DSCFD_UTILS_LOG_GARANTEE_THREAD_SAFE=1 test_log_std_thread_safety.cpp -lpthread -o test_log_std_thread_safety.bin
test_log_std_thread_safety_cuda.bin: test_log_std_thread_safety.cu
	nvcc -std=c++11 -I$(SCFD_INCLUDE) -DSCFD_UTILS_LOG_GARANTEE_THREAD_SAFE=1 test_log_std_thread_safety.cu -lpthread -o test_log_std_thread_safety_cuda.bin