// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.


#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
// #define TEST_HOST
// #define TEST_CUDA
// #define TEST_OPENMP
// #define TEST_UNIFIED_CUDA
// #define TEST_UNIFIED_HOST
// #define TEST_UNIFIED_OPENMP


#include <cstdio>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#ifdef TEST_CUDA
#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#endif
#ifdef TEST_HOST
#include <scfd/memory/host.h>
#include <scfd/for_each/serial_cpu_nd.h>
#endif
#ifdef TEST_OPENMP
#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_nd.h>
#include <scfd/for_each/openmp_nd_impl.h>
#endif
#ifdef TEST_UNIFIED_CUDA
#include <scfd/memory/unified.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#endif
#ifdef TEST_UNIFIED_HOST
#include <scfd/memory/unified.h>
#include <scfd/for_each/serial_cpu_nd.h>
#endif
#ifdef TEST_UNIFIED_OPENMP
#include <scfd/memory/unified.h>
#include <scfd/for_each/openmp_nd.h>
#include <scfd/for_each/openmp_nd_impl.h>
#endif

//#include <scfd/for_each/for_each_storage_types.h>



//TODO restore auto chooser
//static const t_tensor_field_storage     TFS_TYPE = t_for_each_storage_type_helper<FET_TYPE>::storage;

//#define DO_RESULTS_OUTPUT
//#define NDIM          2

#define SZ_X    100
//to make it more stressfull
#define SZ_Y    101
#define SZ_Z    102

#ifdef TEST_CUDA
template<int dim>
using for_each_t = scfd::for_each::cuda_nd<dim>;
using mem_t = scfd::memory::cuda_device;
#endif
#ifdef TEST_HOST
template<int dim>
using for_each_t = scfd::for_each::serial_cpu_nd<dim>;
using mem_t = scfd::memory::host;
#endif
#ifdef TEST_OPENMP
template<int dim>
using for_each_t = scfd::for_each::openmp_nd<dim>;
using mem_t = scfd::memory::host;
#endif
#ifdef TEST_UNIFIED_CUDA
template<int dim>
using for_each_t = scfd::for_each::cuda_nd<dim>;
using mem_t = scfd::memory::unified;
#endif
#ifdef TEST_UNIFIED_HOST
template<int dim>
using for_each_t = scfd::for_each::serial_cpu_nd<dim>;
using mem_t = scfd::memory::unified;
#endif
#ifdef TEST_UNIFIED_OPENMP
template<int dim>
using for_each_t = scfd::for_each::openmp_nd<dim>;
using mem_t = scfd::memory::unified;
#endif



using scfd::static_vec::rect;

typedef scfd::static_vec::vec<int,1>                               t_idx1;
typedef scfd::arrays::tensor0_array_nd<int,1,mem_t>                t_field0_nd1;
typedef scfd::arrays::tensor0_array_nd_view<int,1,mem_t>           t_field0_nd1_view;
typedef scfd::arrays::tensor1_array_nd<int,1,mem_t,3>              t_field1_nd1;
typedef scfd::arrays::tensor1_array_nd_view<int,1,mem_t,3>         t_field1_nd1_view;
typedef scfd::arrays::tensor2_array_nd<int,1,mem_t,3,4>            t_field2_nd1;
typedef scfd::arrays::tensor2_array_nd_view<int,1,mem_t,3,4>       t_field2_nd1_view;

typedef scfd::static_vec::vec<int,2>                               t_idx2;
typedef scfd::arrays::tensor0_array_nd<int,2,mem_t>                t_field0_nd2;
typedef scfd::arrays::tensor0_array_nd_view<int,2,mem_t>           t_field0_nd2_view;
typedef scfd::arrays::tensor1_array_nd<int,2,mem_t,3>              t_field1_nd2;
typedef scfd::arrays::tensor1_array_nd_view<int,2,mem_t,3>         t_field1_nd2_view;
typedef scfd::arrays::tensor2_array_nd<int,2,mem_t,3,4>            t_field2_nd2;
typedef scfd::arrays::tensor2_array_nd_view<int,2,mem_t,3,4>       t_field2_nd2_view;

typedef scfd::static_vec::vec<int,3>                               t_idx3;
typedef scfd::arrays::tensor0_array_nd<int,3,mem_t>                t_field0_nd3;
typedef scfd::arrays::tensor0_array_nd_view<int,3,mem_t>           t_field0_nd3_view;
typedef scfd::arrays::tensor1_array_nd<int,3,mem_t,3>              t_field1_nd3;
typedef scfd::arrays::tensor1_array_nd_view<int,3,mem_t,3>         t_field1_nd3_view;
typedef scfd::arrays::tensor2_array_nd<int,3,mem_t,3,4>            t_field2_nd3;
typedef scfd::arrays::tensor2_array_nd_view<int,3,mem_t,3,4>       t_field2_nd3_view;

struct func_test_field0_nd3
{
    func_test_field0_nd3(const t_field0_nd3 &_f) : f(_f) {}
    t_field0_nd3  f;
    __device__ __host__ void operator()(const t_idx3 &idx)
    {
        f(idx) += 1 - idx[2]*idx[2];
    }
};

bool    test_field0_nd3()
{
    t_field0_nd3          f;
    f.init(t_idx3(SZ_X,SZ_Y,SZ_Z));

    t_field0_nd3_view     view;
    view.init(f, false);
    for (int i = 0;i < SZ_X;++i)
    for (int j = 0;j < SZ_Y;++j)
    for (int k = 0;k < SZ_Z;++k) {
        view(i, j, k) = i + j - k;
    }
    view.release();

    rect<int, 3>              range(t_idx3(0,0,0), t_idx3(SZ_X,SZ_Y,SZ_Z));
    for_each_t<3>             for_each;
    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    for_each.block_size = 128;
    #endif    
    for_each(func_test_field0_nd3(f),range);
    for_each.wait();
    bool    result = true;

    t_field0_nd3_view     view2;
    view2.init(f, true);
    for (int i = 0;i < SZ_X;++i)
    for (int j = 0;j < SZ_Y;++j)
    for (int k = 0;k < SZ_Z;++k) {
        if (view2(i, j, k) != i + j - k + 1 - k*k) {
            printf("test_field0_nd3: i = %d j = %d k = %d: %d != %d \n", i, j, k, view2(i, j, k), i + j - k + 1 - k*k);
            result = false;
        }
        #ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %d\n", i, j, view2(i, j, 0));
        #endif
    }
    view2.release();

    return result;
}


struct func_test_field1_nd2
{
    func_test_field1_nd2(const t_field1_nd2 &_f) : f(_f) {}
    t_field1_nd2  f;
    __device__ __host__ void operator()(const t_idx2 &idx)
    {
        f(idx,0) += 1;
        f(idx,1) -= idx[0];
        //f(idx,2) -= idx[1];
        //different type of indexing
        f(idx[0],idx[1],2) -= idx[1];
    }
};

bool    test_field1_nd2()
{
    t_field1_nd2          f;
    f.init(t_idx2(SZ_X,SZ_Y));

    t_field1_nd2_view     view;
    view.init(f, false);
    for (int i = 0;i < SZ_X;++i)
    for (int j = 0;j < SZ_Y;++j) {
        view(i, j, 0) = i;
        view(i, j, 1) = i+j;
        view(i, j, 2) = i*2+j;
    }
    view.release();

    rect<int, 2>              range(t_idx2(0,0), t_idx2(SZ_X,SZ_Y));
    for_each_t<2>             for_each;
    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    for_each.block_size = 128;
    #endif    
    for_each(func_test_field1_nd2(f),range);
    for_each.wait();
    bool    result = true;

    t_field1_nd2_view     view2;
    view2.init(f, true);
    for (int i = 0;i < SZ_X;++i)
    for (int j = 0;j < SZ_Y;++j) {
        if (view2(i, j, 0) != i+1) {
            printf("test_field1_nd2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 0), i+1);
            result = false;
        }
        if (view2(i, j, 1) != i+j-i) {
            printf("test_field1_nd2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 1), i+j-i);
            result = false;
        }
        if (view2(i, j, 2) != i*2+j-j) {
            printf("test_field1_nd2: i = %d j = %d: %d != %d \n", i, j, view2(i, j, 2), i*2+j-j);
            result = false;
        }
        #ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %d, %d, %d\n", i, j, view2(i, j, 0), view2(i, j, 1), view2(t_idx2(i,j), 2));
        #endif
    }
    view2.release();
    
    return result;
}

int main()
{
    try {

    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    //scfd::utils::init_cuda(-2, 0);
    scfd::utils::init_cuda_persistent();
    #endif
    int err_code = 0;

    if (test_field0_nd3()) {
        printf("test_field0_nd3 seems to be OK\n");
    } else {
        printf("test_field0_nd3 failed\n");
        err_code = 2;
    }

    if (test_field1_nd2()) {
        printf("test_field1_nd2 seems to be OK\n");
    } else {
        printf("test_field1_nd2 failed\n");
        err_code = 2;
    }

    return err_code;
    
    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }
}
