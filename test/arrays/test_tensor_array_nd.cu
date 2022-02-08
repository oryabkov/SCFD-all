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

#include <stdexcept>
#include <string>
#include <array>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/static_vec/vec.h>
#include <scfd/memory/host.h>
#include <scfd/memory/cuda.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

//#define DO_RESULTS_OUTPUT

typedef scfd::utils::cuda_timer_event                                     timer_event_t;

typedef scfd::memory::cuda_device                                         mem_t;
typedef int                                                               t_idx;
typedef scfd::static_vec::vec<float,3>                                    t_vec3;
typedef scfd::arrays::tensor0_array<float,mem_t>                          array0_t;
typedef scfd::arrays::tensor0_array_view<float,mem_t>                     array0_view_t;
typedef scfd::arrays::tensor1_array<float,mem_t,3>                        array1_t;
typedef array1_t::view_type                                               array1_view_t;
typedef scfd::arrays::tensor2_array<float,mem_t,2,3>                      array2_t;
typedef array2_t::view_type                                               array2_view_t;
typedef scfd::arrays::tensor3_array<float,mem_t,2,4,3>                    array3_t;
typedef array3_t::view_type                                               array3_view_t;
typedef scfd::arrays::tensor4_array<float,mem_t,2,4,5,3>                  array4_t;
typedef array4_t::view_type                                               array4_view_t;

#define NDIM                                                              2
typedef scfd::static_vec::vec<int,NDIM>                                   idx_nd_t;
typedef scfd::arrays::tensor1_array_nd<float,NDIM,mem_t,3>                array1_nd_t;
typedef array1_nd_t::view_type                                            array1_nd_view_t;
typedef scfd::arrays::tensor1_array_nd<float,NDIM,mem_t,scfd::arrays::dyn_dim>  array1_nd_dyn_t;
typedef array1_nd_dyn_t::view_type                                        array1_nd_dyn_view_t;

//global test array size parameters
int     sz1 = 100, sz2 = 100, reps_n = 10;
int     block_sz1 = 256;
int     block2_sz1 = 16,block2_sz2 = 16;

__global__ void test_ker_array0(array0_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.get_dim<0>())) return;
    f(i) += i;
}

__global__ void test_ker_ptr0(float *data, int sz1)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < sz1)) return;
    data[i] += i;
}

__global__ void test_ker_array1(array1_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.get_dim<0>())) return;
    f(i, 0) += i;
    f(i, 1) += i;
    f(i, 2) -= i;

    //t_vec3  v = f.get<t_vec3>(i);
    t_vec3  v;
    f.get_vec(v,i);
    //v = f.getv(i);
}

__global__ void test_ker_array1_vec(array1_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.size())) return;
    float v[3];
    f.get_vec(v,i);
    v[0] += i;
    v[1] += i;
    v[2] -= i;
    f.set_vec(v,i);
}

__global__ void test_ker_ptr1(float *data, int sz1)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < sz1)) return;
    data[i + 0*sz1] += i;
    data[i + 1*sz1] += i;
    data[i + 2*sz1] -= i;
}

__global__ void test_ker_array2(array2_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x - 1;
    if (!(i+1 < f.get_dim<0>())) return;
    f(i, 0, 0) += i;
    f(i, 0, 1) += i;
    f(i, 0, 2) -= i;
    f(i, 1, 0) += i;
    f(i, 1, 1) += i;
    f(i, 1, 2) -= i;

    //t_vec3  v = f.get<t_vec3>(i);
    t_vec3  v;
    f.get_vec(v, i, 0);
    //v = f.getv(i);
}

__global__ void test_ker_ptr2(float *data, int sz1)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x - 1;
    if (!(i+1 < sz1)) return;
    data[i+1 + 0*sz1 + 0*sz1*2] += i;
    data[i+1 + 0*sz1 + 1*sz1*2] += i;
    data[i+1 + 0*sz1 + 2*sz1*2] -= i;
    data[i+1 + 1*sz1 + 0*sz1*2] += i;
    data[i+1 + 1*sz1 + 1*sz1*2] += i;
    data[i+1 + 1*sz1 + 2*sz1*2] -= i;
}

__global__ void test_ker_array3(array3_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.get_dim<0>())) return;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2) {
        f(i, i1, i2, 0) += i*(i1+1)+i2;
        f(i, i1, i2, 1) += i*(i1+1)+i2;
        f(i, i1, i2, 2) -= i*(i1+1)+i2;
    }

    //t_vec3  v = f.get<t_vec3>(i);
    t_vec3  v;
    f.get_vec(v, i, 0, 0);
    //v = f.getv(i);
}

__global__ void test_ker_ptr3(float *data, int sz1)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < sz1)) return;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2) {
        data[i + i1*sz1 + i2*sz1*2 + 0*sz1*2*4] += i*(i1+1)+i2;
        data[i + i1*sz1 + i2*sz1*2 + 1*sz1*2*4] += i*(i1+1)+i2;
        data[i + i1*sz1 + i2*sz1*2 + 2*sz1*2*4] -= i*(i1+1)+i2;
    }
}

__global__ void test_ker_array4(array4_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.get_dim<0>())) return;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2)
    for (int i3 = 0;i3 < 5;++i3) {
        f(i, i1, i2, i3, 0) += i*(i1+1)+(i2*5)/(i3+1);
        f(i, i1, i2, i3, 1) += i*(i1+1)+(i2*5)/(i3+1);
        f(i, i1, i2, i3, 2) -= i*(i1+1)+(i2*5)/(i3+1);
    }

    //t_vec3  v = f.get<t_vec3>(i);
    t_vec3  v;
    f.get_vec(v, i, 0, 0, 0);
    //v = f.getv(i);
}

__global__ void test_ker_ptr4(float *data, int sz1)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < sz1)) return;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2)
    for (int i3 = 0;i3 < 5;++i3) {
        data[i + i1*sz1 + i2*sz1*2 + i3*sz1*2*4 + 0*sz1*2*4*5] += i*(i1+1)+(i2*5)/(i3+1);
        data[i + i1*sz1 + i2*sz1*2 + i3*sz1*2*4 + 1*sz1*2*4*5] += i*(i1+1)+(i2*5)/(i3+1);
        data[i + i1*sz1 + i2*sz1*2 + i3*sz1*2*4 + 2*sz1*2*4*5] -= i*(i1+1)+(i2*5)/(i3+1);
    }
}

__global__ void test_ker_array1_nd_1(array1_nd_t f)
{
    int     i1 = blockIdx.x * blockDim.x + threadIdx.x + f.get_index0<0>(),
            i2 = blockIdx.y * blockDim.y + threadIdx.y + f.get_index0<1>();
    if (!((i1-f.get_index0<0>() < f.get_dim<0>())&&
          (i2-f.get_index0<1>() < f.get_dim<1>()))) return;
    f(idx_nd_t(i1,i2),0) += 1;
    f(idx_nd_t(i1,i2),1) -= i1;
    f(idx_nd_t(i1,i2),2) -= i2;
    
    //t_vec3        v = f.getv<t_vec3>(t_idx(i1,i2));
    //t_vec3  v = f.get<t_vec3>(t_idx(i1,i2));
    t_vec3  v;
    //f.get_vec(v, idx_nd_t(i1,i2));
    f.get_vec(v, i1, i2);
    //v = f.getv(t_idx(i1,i2));
}

__global__ void test_ker_ptr1_nd(float *data, int sz1, int sz2, int shift1, int shift2)
{
    int     i1 = blockIdx.x * blockDim.x + threadIdx.x + shift1,
            i2 = blockIdx.y * blockDim.y + threadIdx.y + shift2;
    if (!((i1-shift1 < sz1)&&(i2-shift2 < sz2))) return;
    data[i1-shift1 + (i2-shift2)*sz1 + 0*sz1*sz2] += 1;
    data[i1-shift1 + (i2-shift2)*sz1 + 1*sz1*sz2] -= i1;
    data[i1-shift1 + (i2-shift2)*sz1 + 2*sz1*sz2] -= i2;
}

__global__ void test_ker_array1_nd_2(array1_nd_dyn_t f)
{
    int     i1 = blockIdx.x * blockDim.x + threadIdx.x + f.get_index0<0>(),
            i2 = blockIdx.y * blockDim.y + threadIdx.y + f.get_index0<1>();
    if (!((i1-f.get_index0<0>() < f.get_dim<0>())&&
          (i2-f.get_index0<1>() < f.get_dim<1>()))) return;
    f(idx_nd_t(i1,i2),0) += 1;
    f(idx_nd_t(i1,i2),1) -= i1;
    f(idx_nd_t(i1,i2),2) -= i2;
    
    //t_vec3        v = f.getv<t_vec3>(t_idx(i1,i2));
    //t_vec3  v = f.get<t_vec3>(t_idx(i1,i2));
    t_vec3  v;
    //f.get_vec(v, idx_nd_t(i1,i2));
    //f.get_vec(v, i1, i2);
    //v = f.getv(t_idx(i1,i2));
}

bool    test_array0()
{
    array0_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1, 0);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array0<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array0: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr0<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array0: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    array0_view_t   view(f, false);
    for (int i = 0;i < sz1;++i) {
        view(i) = 1;
    }
    view.release();

    //test correctness
    e1.record();
    test_ker_array0<<<dimGrid, dimBlock>>>(f);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    e2.record();

    bool            result = true;

    auto            view2 = f.create_view(true);
    for (int i = 0;i < sz1;++i) {
        if (view2(i) != 1+i) {
            printf("test_array0: i = %d: %f != %f \n", i, view2(i), float(1+i));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f\n", i, view2(i));
#endif
    }
    view2.release();

    return result;
}

bool    test_array1()
{
    array1_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array1<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr1<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    auto            view = f.create_view(false);
    for (int i = 0;i < sz1;++i) {
        view(i, 0) = 1;
        view(i, 1) = i;
        view(i, 2) = i;
    }
    view.release();

    //call some kernel
    test_ker_array1<<<dimGrid, dimBlock>>>(f);
    
    bool    result = true;

    array1_view_t   view2 = f.create_view(true);
    for (int i = 0;i < sz1;++i) {
        if (view2(i, 0) != 1+i) {
            printf("test_array1: i = %d, i0 = 0: %f != %f \n", i, view2(i, 0), float(1+i));
            result = false;
        }
        if (view2(i, 1) != i+i) {
            printf("test_array1: i = %d, i0 = 1: %f != %f \n", i, view2(i, 1), float(i+i));
            result = false;
        }
        if (view2(i, 2) != i-i) {
            printf("test_array1: i = %d, i0 = 2: %f != %f \n", i, view2(i, 2), float(i-i));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f, %f, %f\n", i, view2(i, 0), view2(i, 1), view2(i, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_array1_vec()
{
    array1_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array1_vec<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_vec: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr1<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_vec: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    auto            view = f.create_view(false);
    for (int i = 0;i < sz1;++i) {
        view(i, 0) = 1;
        view(i, 1) = i;
        view(i, 2) = i;
    }
    view.release();

    //call some kernel
    test_ker_array1_vec<<<dimGrid, dimBlock>>>(f);
    
    bool    result = true;

    array1_view_t   view2 = f.create_view(true);
    for (int i = 0;i < sz1;++i) {
        if (view2(i, 0) != 1+i) {
            printf("test_array1_vec: i = %d, i0 = 0: %f != %f \n", i, view2(i, 0), float(1+i));
            result = false;
        }
        if (view2(i, 1) != i+i) {
            printf("test_array1_vec: i = %d, i0 = 1: %f != %f \n", i, view2(i, 1), float(i+i));
            result = false;
        }
        if (view2(i, 2) != i-i) {
            printf("test_array1_vec: i = %d, i0 = 2: %f != %f \n", i, view2(i, 2), float(i-i));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f, %f, %f\n", i, view2(i, 0), view2(i, 1), view2(i, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_array2()
{
    array2_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1,-1);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array2<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array2: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr2<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array2: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    array2_view_t   view(f, false);
    for (int i = -1;i < sz1-1;++i) {
        view(i, 0, 0) = 1;
        view(i, 0, 1) = i;
        view(i, 0, 2) = i;

        view(i, 1, 0) = 1+2;
        view(i, 1, 1) = i+2;
        view(i, 1, 2) = i+2;
    }
    view.release();

    //call some kernel
    test_ker_array2<<<dimGrid, dimBlock>>>(f);

    bool    result = true;

    auto            view2 = f.create_view(true);
    for (int i = -1;i < sz1-1;++i) {
        if (view2(i, 0, 0) != 1+i) {
            printf("test_array2: i = %d, i0 = 0, i1 = 0: %f != %f \n", i, view2(i, 0, 0), float(1+i));
            result = false;
        }
        if (view2(i, 0, 1) != i+i) {
            printf("test_array2: i = %d, i0 = 0, i1 = 1: %f != %f \n", i, view2(i, 0, 1), float(i+i));
            result = false;
        }
        if (view2(i, 0, 2) != i-i) {
            printf("test_array2: i = %d, i0 = 0, i1 = 2: %f != %f \n", i, view2(i, 0, 2), float(i-i));
            result = false;
        }
        if (view2(i, 1, 0) != 1+i+2) {
            printf("test_array2: i = %d, i0 = 1, i1 = 0: %f != %f \n", i, view2(i, 1, 0), float(1+i+2));
            result = false;
        }
        if (view2(i, 1, 1) != i+i+2) {
            printf("test_array2: i = %d, i0 = 1, i1 = 1: %f != %f \n", i, view2(i, 1, 1), float(i+i+2));
            result = false;
        }
        if (view2(i, 1, 2) != i-i+2) {
            printf("test_array2: i = %d, i0 = 1, i1 = 2: %f != %f \n", i, view2(i, 1, 2), float(i-i+2));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f, %f, %f, %f, %f, %f\n", i, view2(i, 0, 0), view2(i, 0, 1), view2(i, 0, 2),  view2(i, 1, 0), view2(i, 1, 1), view2(i, 1, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_array3()
{
    array3_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array3<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array3: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr3<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array3: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    auto            view = f.create_view(false);
    for (int i = 0;i < sz1;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2) {
            view(i, i1, i2, 0) = 1*(i2+1)+i1;
            view(i, i1, i2, 1) = i*(i2+1)+i1;
            view(i, i1, i2, 2) = i*(i2+1)+i1;
        }
    }
    view.release();

    //call some kernel
    test_ker_array3<<<dimGrid, dimBlock>>>(f);

    bool    result = true;

    auto            view2 = f.create_view(true);
    for (int i = 0;i < sz1;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2) {
            if (view2(i, i1, i2, 0) != 1*(i2+1)+i1+(i*(i1+1)+i2)) {
                printf("test_array3: i = %d, i1 = %d, i2 = %d: %f != %f \n", i, i1, i2, view2(i, i1, i2, 0), float(1*(i2+1)+i1+(i*(i1+1)+i2)));
                result = false;
            }
            if (view2(i, i1, i2, 1) != i*(i2+1)+i1+(i*(i1+1)+i2)) {
                printf("test_array3: i = %d, i1 = %d, i2 = %d: %f != %f \n", i, i1, i2, view2(i, i1, i2, 1), float(i*(i2+1)+i1+(i*(i1+1)+i2)));
                result = false;
            }
            if (view2(i, i1, i2, 2) != i*(i2+1)+i1-(i*(i1+1)+i2)) {
                printf("test_array3: i = %d, i1 = %d, i2 = %d: %f != %f \n", i, i1, i2, view2(i, i1, i2, 2), float(i*(i2+1)+i1-(i*(i1+1)+i2)));
                result = false;
            }
        }

#ifdef DO_RESULTS_OUTPUT
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2)
            printf("%d, %f, %f, %f\n", i, view2(i, i1, i2, 0), view2(i, i1, i2, 1), view2(i, i1, i2, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_array4()
{
    array4_t            f;
    timer_event_t       e1,e2,e3,e4;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array4<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array4: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr4<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array4: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    array4_view_t   view(f, false);
    for (int i = 0;i < sz1;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2)
        for (int i3 = 0;i3 < 5;++i3) {
            view(i, i1, i2, i3, 0) = 1*(i2+1)+i1+i3;
            view(i, i1, i2, i3, 1) = i*(i2+1)+i1+i3;
            view(i, i1, i2, i3, 2) = i*(i2+1)+i1+i3*2;
        }
    }
    view.release();

    //call some kernel
    test_ker_array4<<<dimGrid, dimBlock>>>(f);

    bool    result = true;

    array4_view_t   view2(f, true);
    for (int i = 0;i < sz1;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2)
        for (int i3 = 0;i3 < 5;++i3) {
            if (view2(i, i1, i2, i3, 0) != 1*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_array4: i = %d, i1 = %d, i2 = %d, i3 = %d: %f != %f \n", i, i1, i2, i3, view2(i, i1, i2, i3, 0), float(1*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))));
                result = false;
            }
            if (view2(i, i1, i2, i3, 1) != i*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_array4: i = %d, i1 = %d, i2 = %d, i3 = %d: %f != %f \n", i, i1, i2, i3, view2(i, i1, i2, i3, 1), float(i*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))));
                result = false;
            }
            if (view2(i, i1, i2, i3, 2) != i*(i2+1)+i1+i3*2-(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_array4: i = %d, i1 = %d, i2 = %d, i3 = %d: %f != %f \n", i, i1, i2, i3, view2(i, i1, i2, i3, 2), float(i*(i2+1)+i1+i3*2-(i*(i1+1)+(i2*5)/(i3+1))));
                result = false;
            }
        }

#ifdef DO_RESULTS_OUTPUT
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2)
        for (int i3 = 0;i3 < 5;++i3)
            printf("%d, %f, %f, %f\n", i, view2(i, i1, i2, i3, 0), view2(i, i1, i2, i3, 1), view2(i, i1, i2, i3, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_assign_operator()
{
    array0_t        f0,f0_;
    array1_t        f1,f1_;
    array2_t        f2,f2_;
    array2_t        f3,f3_;
    array2_t        f4,f4_;
    f0.init(sz1);
    f1.init(sz1);
    f2.init(sz1);
    f3.init(sz1);
    f4.init(sz1);
    f0_ = f0;
    f1_ = f1;
    f2_ = f2;
    f3_ = f3;
    f4_ = f4;

    if ((!f0.is_free())&&(!f0_.is_free())&&
        (!f1.is_free())&&(!f1_.is_free())&&
        (!f2.is_free())&&(!f2_.is_free())&&
        (!f3.is_free())&&(!f3_.is_free())&&
        (!f4.is_free())&&(!f4_.is_free())) return true;

    return false;
}

bool    test_array1_nd_1(int shift1, int shift2)
{
    array1_nd_t     f;
    timer_event_t   e1,e2,e3,e4;
    dim3            dimBlock(block2_sz1,block2_sz2);
    dim3            dimGrid((sz1/block2_sz1)+1,(sz2/block2_sz2)+1);
    if ((shift1 == 0)&&(shift2 == 0))
        f.init(idx_nd_t(sz1,sz2));
    else
        f.init(idx_nd_t(sz1,sz2), idx_nd_t(shift1,shift2));

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array1_nd_1<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_nd_1: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr1_nd<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1, sz2, shift1, shift2);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_nd_1: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    array1_nd_view_t    view(f, false);
    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        view(idx_nd_t(i,j), 0) = i;
        view(idx_nd_t(i,j), 1) = i+j;
        view(idx_nd_t(i,j), 2) = i*2+j;
    }
    view.release();

    //call some kernel
    test_ker_array1_nd_1<<<dimGrid, dimBlock>>>(f);

    bool    result = true;

    array1_nd_view_t    view2(f, true);
    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        if (view2(idx_nd_t(i,j), 0) != i+1) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 0: %f != %f \n", i, j, view2(idx_nd_t(i,j), 0), float(i+1));
            result = false;
        }
        if (view2(idx_nd_t(i,j), 1) != i+j-i) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 1: %f != %f \n", i, j, view2(idx_nd_t(i,j), 1), float(i+j-i));
            result = false;
        }
        if (view2(idx_nd_t(i,j), 2) != i*2+j-j) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 2: %f != %f \n", i, j, view2(idx_nd_t(i,j), 2), float(i*2+j-j));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %f, %f, %f\n", i, j, view2(idx_nd_t(i,j), 0), view2(idx_nd_t(i,j), 1), view2(idx_nd_t(i,j), 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_array1_nd_2(int shift1, int shift2)
{
    array1_nd_dyn_t f;
    timer_event_t   e1,e2,e3,e4;
    dim3            dimBlock(block2_sz1,block2_sz2);
    dim3            dimGrid((sz1/block2_sz1)+1,(sz2/block2_sz2)+1);
    if ((shift1 == 0)&&(shift2 == 0))
        f.init(idx_nd_t(sz1,sz2),3);
    else
        f.init(idx_nd_t(sz1,sz2), idx_nd_t(shift1,shift2), 3, 0);

    //test time with array
    e1.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_array1_nd_2<<<dimGrid, dimBlock>>>(f);
    }
    e2.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_nd_2: array time = %f ms\n", e2.elapsed_time(e1)/reps_n);

    //test time with c kernel
    e3.record();
    for (int rep = 0;rep < reps_n;++rep) {
        test_ker_ptr1_nd<<<dimGrid, dimBlock>>>(f.raw_ptr(), sz1, sz2, shift1, shift2);
    }
    e4.record();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("test_array1_nd_2: plain c kernel time = %f ms\n", e4.elapsed_time(e3)/reps_n);

    array1_nd_dyn_view_t    view(f, false);
    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        view(idx_nd_t(i,j), 0) = i;
        view(idx_nd_t(i,j), 1) = i+j;
        view(idx_nd_t(i,j), 2) = i*2+j;
    }
    view.release();

    //call some kernel
    test_ker_array1_nd_2<<<dimGrid, dimBlock>>>(f);

    bool    result = true;

    array1_nd_dyn_view_t    view2(f, true);
    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        if (view2(idx_nd_t(i,j), 0) != i+1) {
            printf("test_array1_nd_2: i = %d, j = %d, i0 = 0: %f != %f \n", i, j, view2(idx_nd_t(i,j), 0), float(i+1));
            result = false;
        }
        if (view2(idx_nd_t(i,j), 1) != i+j-i) {
            printf("test_array1_nd_2: i = %d, j = %d, i0 = 1: %f != %f \n", i, j, view2(idx_nd_t(i,j), 1), float(i+j-i));
            result = false;
        }
        if (view2(idx_nd_t(i,j), 2) != i*2+j-j) {
            printf("test_array1_nd_2: i = %d, j = %d, i0 = 2: %f != %f \n", i, j, view2(idx_nd_t(i,j), 2), float(i*2+j-j));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %f, %f, %f\n", i, j, view2(idx_nd_t(i,j), 0), view2(idx_nd_t(i,j), 1), view2(idx_nd_t(i,j), 2));
#endif
    }
    view2.release();

    return result;
}

int main(int argc, char **args)
{
    try {

    if (argc < 4) {
        printf("USAGE: %s <size1> <size2> <reps_num>\n", args[0]);
        //printf("standart test 100x100\n");
        return 0;
    } else {
        sz1 = std::stoi(args[1]);
        sz2 = std::stoi(args[2]);
        reps_n = std::stoi(args[3]);
    }

    int err_code = 0;

    scfd::utils::init_cuda(-2, 0);

    if (test_array0()) {
        printf("test_array0 seems to be OK\n");
    } else {
        printf("test_array0 failed\n");
        err_code = 2;
    }

    if (test_array1()) {
        printf("test_array1 seems to be OK\n");
    } else {
        printf("test_array1 failed\n");
        err_code = 2;
    }

    if (test_array1_vec()) {
        printf("test_array1_vec seems to be OK\n");
    } else {
        printf("test_array1_vec failed\n");
        err_code = 2;
    }
    
    if (test_array2()) {
        printf("test_array2 seems to be OK\n");
    } else {
        printf("test_array2 failed\n");
        err_code = 2;
    }
    
    if (test_array3()) {
        printf("test_array3 seems to be OK\n");
    } else {
        printf("test_array3 failed\n");
        err_code = 2;
    }
    
    if (test_array4()) {
        printf("test_array4 seems to be OK\n");
    } else {
        printf("test_array4 failed\n");
        err_code = 2;
    }
    
    if (test_assign_operator()) {
        printf("test_assign_operator seems to be OK\n");
    } else {
        printf("test_assign_operator failed\n");
        err_code = 2;
    }

    if (test_array1_nd_1(1,1)) {
        printf("test_array1_nd_1 seems to be OK\n");
    } else {
        printf("test_array1_nd_1 failed\n");
        err_code = 2;
    }

    if (test_array1_nd_2(0,0)) {
        printf("test_array1_nd_2 seems to be OK\n");
    } else {
        printf("test_array1_nd_2 failed\n");
        err_code = 2;
    }

    return err_code;

    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }
}