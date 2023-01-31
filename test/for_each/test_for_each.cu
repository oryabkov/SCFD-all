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
#ifdef TEST_CUDA
#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda.h>
#include <scfd/for_each/cuda_impl.cuh>
#endif
#ifdef TEST_HOST
#include <scfd/memory/host.h>
#include <scfd/for_each/serial_cpu.h>
#endif
#ifdef TEST_OPENMP
#include <scfd/memory/host.h>
#include <scfd/for_each/openmp.h>
#include <scfd/for_each/openmp_impl.h>
#endif
#ifdef TEST_UNIFIED_CUDA
#include <scfd/memory/unified.h>
#include <scfd/for_each/cuda.h>
#include <scfd/for_each/cuda_impl.cuh>
#endif
#ifdef TEST_UNIFIED_HOST
#include <scfd/memory/unified.h>
#include <scfd/for_each/serial_cpu.h>
#endif
#ifdef TEST_UNIFIED_OPENMP
#include <scfd/memory/unified.h>
#include <scfd/for_each/openmp.h>
#include <scfd/for_each/openmp_impl.h>
#endif
//#include <scfd/for_each/for_each_storage_types.h>



//TODO restore auto chooser
//static const t_tensor_field_storage     TFS_TYPE = t_for_each_storage_type_helper<FET_TYPE>::storage;

//#define DO_RESULTS_OUTPUT
//#define NDIM          2

#define SZ_X    100

#ifdef TEST_CUDA
using for_each_t = scfd::for_each::cuda<>;
using mem_t = scfd::memory::cuda_device;
#endif
#ifdef TEST_HOST
using for_each_t = scfd::for_each::serial_cpu<>;
using mem_t = scfd::memory::host;
#endif
#ifdef TEST_OPENMP
using for_each_t = scfd::for_each::openmp<>;
using mem_t = scfd::memory::host;
#endif
#ifdef TEST_UNIFIED_CUDA
using for_each_t = scfd::for_each::cuda<>;
using mem_t = scfd::memory::unified;
#endif
#ifdef TEST_UNIFIED_HOST
using for_each_t = scfd::for_each::serial_cpu<>;
using mem_t = scfd::memory::unified;
#endif
#ifdef TEST_UNIFIED_OPENMP
using for_each_t = scfd::for_each::openmp<>;
using mem_t = scfd::memory::unified;
#endif

//using scfd::static_vec::rect;

typedef scfd::arrays::tensor0_array<int,mem_t>                t_field0;
typedef scfd::arrays::tensor0_array_view<int,mem_t>           t_field0_view;
typedef scfd::arrays::tensor1_array<int,mem_t,3>              t_field1;
typedef scfd::arrays::tensor1_array_view<int,mem_t,3>         t_field1_view;
typedef scfd::arrays::tensor2_array<int,mem_t,3,4>            t_field2;
typedef scfd::arrays::tensor2_array_view<int,mem_t,3,4>       t_field2_view;

/*typedef scfd::static_vec::vec<int,2>                               t_idx2;
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
typedef scfd::arrays::tensor2_array_nd_view<int,3,mem_t,3,4>       t_field2_nd3_view;*/

struct func_test_field0
{
    func_test_field0(const t_field0 &f_) : f(f_) {}
    t_field0  f;
    __device__ __host__ void operator()(const int &idx)
    {
        f(idx) += 1 - idx*idx;
    }
};

bool    test_field0()
{
    t_field0          f;
    f.init(SZ_X);

    t_field0_view     view;
    view.init(f, false);
    for (int i = 0;i < SZ_X;++i)
    {
        view(i) = i;
    }
    view.release();

    for_each_t             for_each;
    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    for_each.block_size = 128;
    #endif
    for_each(func_test_field0(f), 0, SZ_X);
    for_each.wait();
    bool    result = true;

    t_field0_view     view2;
    view2.init(f, true);
    for (int i = 0;i < SZ_X;++i)
    {
        if (view2(i) != i + 1 - i*i) 
        {
            printf("test_field0: i = %d: %d != %d \n", i, view2(i), i + 1 - i*i);
            result = false;
        }
        #ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %d\n", i, view2(i));
        #endif
    }
    view2.release();

    return result;
}


struct func_test_field1
{
    func_test_field1(const t_field1 &f_) : f(f_) {}
    t_field1  f;
    __device__ __host__ void operator()(const int &idx)
    {
        f(idx,0) += 1;
        f(idx,1) -= idx;
        f(idx,2) -= idx;
    }
};

bool    test_field1()
{
    t_field1          f;
    f.init(SZ_X);

    t_field1_view     view;
    view.init(f, false);
    for (int i = 0;i < SZ_X;++i)
    {
        view(i, 0) = i;
        view(i, 1) = i;
        view(i, 2) = i*2;
    }
    view.release();

    for_each_t                for_each;
    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    for_each.block_size = 128;
    #endif
    for_each(func_test_field1(f), 0, SZ_X);
    for_each.wait();
    bool    result = true;

    t_field1_view     view2;
    view2.init(f, true);
    for (int i = 0;i < SZ_X;++i)
    {
        if (view2(i, 0) != i+1) 
        {
            printf("test_field1: i = %d: %d != %d \n", i, view2(i, 0), i+1);
            result = false;
        }
        if (view2(i, 1) != i-i) 
        {
            printf("test_field1: i = %d: %d != %d \n", i, view2(i, 1), i-i);
            result = false;
        }
        if (view2(i, 2) != i*2-i) 
        {
            printf("test_field1: i = %d: %d != %d \n", i, view2(i, 2), i*2-i);
            result = false;
        }
        #ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %d, %d\n", i, view2(i, 0), view2(i, 1), view2(i, 2));
        #endif
    }
    view2.release();
    
    return result;
}

int main()
{
    try {

    #if defined(TEST_CUDA)||defined(TEST_UNIFIED_CUDA)
    // scfd::utils::init_cuda(-2, 0);
    scfd::utils::init_cuda_persistent();
    #endif
    int err_code = 0;

    if (test_field0()) 
    {
        printf("test_field0 seems to be OK\n");
    } 
    else 
    {
        printf("test_field0 failed\n");
        err_code = 2;
    }

    if (test_field1()) 
    {
        printf("test_field1 seems to be OK\n");
    } 
    else 
    {
        printf("test_field1 failed\n");
        err_code = 2;
    }

    return err_code;
    
    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }
}
