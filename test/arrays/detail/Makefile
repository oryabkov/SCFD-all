SCFD_INCLUDE=../../../include
test_dyn_dim_counter.bin: test_dyn_dim_counter.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_dyn_dim_counter.cpp -o test_dyn_dim_counter.bin
test_template_indexer.bin: test_template_indexer.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_template_indexer.cpp -o test_template_indexer.bin
test_dim_getter.bin: test_dim_getter.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_dim_getter.cpp -o test_dim_getter.bin
test_parameter_indexer.bin: test_parameter_indexer.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_parameter_indexer.cpp -o test_parameter_indexer.bin
test_size_calculator.bin: test_size_calculator.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_size_calculator.cpp -o test_size_calculator.bin
test_template_arg_search.bin: test_template_arg_search.cpp
	g++ -O3 -I$(SCFD_INCLUDE) -std=c++11 test_template_arg_search.cpp -o test_template_arg_search.bin

all: test_dyn_dim_counter.bin test_template_indexer.bin test_dim_getter.bin test_parameter_indexer.bin test_size_calculator.bin test_template_arg_search.bin