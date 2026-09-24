#!/usr/bin/env bash
# Direct-run manifest for existing CMake targets. Keep this list in sync with
# test/*/CMakeLists.txt. The caller provides run_test and run_mpi_test, including
# per-case working directories, logging, timeouts, and failure aggregation.

scfd_run_host_tests() {
    local name
    for name in test_tensor_host test_tensor_array_nd_syntax_host \
                test_last_index_fast_arranger_host test_custom_index_arranger_host; do
        run_test arrays "$name" "test/arrays/$name.bin"
    done
    run_test arrays test_tensor_array_nd_host test/arrays/test_tensor_array_nd_host.bin 100000
    for name in test_template_indexer test_template_arg_search test_size_calculator \
                test_parameter_indexer test_dyn_dim_counter test_dim_getter; do
        run_test arrays "$name" "test/arrays/detail/$name.bin"
    done
    for name in test_vec_host test_rect_host; do
        run_test static_vec "$name" "test/static_vec/$name.bin"
    done
    run_test static_mat test_mat_host test/static_mat/test_mat_host.bin
    run_test for_each test_for_each_func_macro test/for_each/test_for_each_func_macro.bin
    for name in intersect_triangles_test static_vec_traits_test; do
        run_test geometry "$name" "test/geometry/$name.bin"
    done
    run_test for_paper test_omp test/for_paper/test_omp.bin
    for name in test_trivial_comm test_platform_trivial_comm_serial \
                test_platform_runtime_serial test_platform_runtime_omp; do
        run_test communication "$name" "test/communication/$name.bin"
    done
}

scfd_run_cuda_tests() {
    local name variant dimensions
    for name in test_cuda_unified_arrays test_tensor_array_nd_visible_cuda \
                test_custom_index_arranger_cuda; do
        run_test arrays "$name" "test/arrays/$name.bin"
    done
    # Arguments are size1, size2, repetitions; keep the 2D allocation bounded.
    run_test arrays test_tensor_array_nd_cuda test/arrays/test_tensor_array_nd_cuda.bin 1024 32 2
    run_test static_vec test_vec_cuda test/static_vec/test_vec_cuda.bin
    run_test static_mat test_mat_cuda test/static_mat/test_mat_cuda.bin
    # These host/OpenMP variants currently also require CUDA to build their .cu sources.
    for variant in host cuda omp unified_host unified_cuda unified_omp; do
        for name in "test_for_each_$variant" "test_for_each_nd_$variant"; do
            run_test for_each "$name" "test/for_each/$name.bin"
        done
    done
    run_test for_paper test_tensor_array_3d_cross_product \
        test/for_paper/test_tensor_array_3d_cross_product.bin 1024 2 a
    run_test for_paper test_omp_cu test/for_paper/test_omp_cu.bin
    for dimensions in 2 3 4 5; do
        name="test_matmul_cuda_${dimensions}x${dimensions}"
        run_test matmul "$name" "test/matmul/$name.bin" 1024 2 a
    done
    run_test external_libraries test_cusolver_wrap test/external_libraries/test_cusolver_wrap.bin
}

scfd_run_hip_tests() {
    local name dimensions
    run_test for_each test_for_each_hip test/for_each/test_for_each_hip.bin
    run_test for_paper test_hip test/for_paper/test_hip.bin 1024 2 a
    for dimensions in 2 3 4 5; do
        name="test_matmul_hip_${dimensions}x${dimensions}"
        run_test matmul "$name" "test/matmul/$name.bin" 1024 2 a
    done
}

scfd_run_sycl_tests() {
    run_test for_each test_for_each_sycl test/for_each/test_for_each_sycl.bin
    run_test for_each test_for_each_nd_sycl test/for_each/test_for_each_nd_sycl.bin
    run_test for_paper test_sycl test/for_paper/test_sycl.bin 1024 2
}

scfd_run_mpi_tests() {
    local name platform operation level ranks
    for platform in serial omp; do
        name="test_platform_runtime_${platform}_mpi"
        run_mpi_test communication "$name" 4 "test/communication/$name.bin"
        for operation in rect_distributor rect_distributor_tensor; do
            name="test_platform_${operation}_${platform}_mpi"
            run_mpi_test communication "$name" 3 "test/communication/$name.bin" 1 1 1 0 0
        done
    done
    for name in test_sequential_mpi test_mpi_binary_file test_mpi_comm_collectives \
                test_mpi_comm_p2p test_mpi_comm_calls; do
        run_mpi_test communication "$name" 4 "test/communication/$name.bin"
    done
    run_mpi_test communication test_mpi_thread_multiple_default 4 \
        test/communication/test_mpi_thread_multiple.bin
    for level in 0 1 2 3; do
        run_mpi_test communication "test_mpi_thread_multiple_$level" 4 \
            test/communication/test_mpi_thread_multiple.bin "$level"
    done
    for ranks in 3 4; do
        run_mpi_test communication "test_mpi_comm_split_$ranks" "$ranks" \
            test/communication/test_mpi_comm_split.bin
    done
    for name in test_mpi_rect_distributor test_mpi_rect_distributor_tensor; do
        run_mpi_test communication "$name" 3 "test/communication/$name.bin" 1 1 1 0 0
    done
}
