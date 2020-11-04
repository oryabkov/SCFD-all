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
#include <array>
#include <scfd/static_vec/vec.h>
#include <scfd/memory/host.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/last_index_fast_arranger.h>

//#define DO_RESULTS_OUTPUT

typedef scfd::memory::host                                      mem_t;
typedef int                                                     t_idx;
typedef scfd::static_vec::vec<float,3>                          t_vec3;
typedef scfd::arrays::tensor0_array<float,mem_t>                t_field0;
typedef t_field0::view_type                                     t_field0_view;
typedef scfd::arrays::tensor1_array<float,mem_t,3>              t_field1;
typedef t_field1::view_type                                     t_field1_view;
typedef scfd::arrays::tensor2_array<float,mem_t,2,3>            t_field2;
typedef t_field2::view_type                                     t_field2_view;
typedef scfd::arrays::tensor3_array<float,mem_t,2,4,3>          t_field3;
typedef t_field3::view_type                                     t_field3_view;
typedef scfd::arrays::tensor4_array<float,mem_t,2,4,5,3>        t_field4;
typedef t_field4::view_type                                     t_field4_view;

/*__global__ void test_ker_field0(t_field0 f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    f(i) += i;
}

__global__ void test_ker_field1(t_field1 f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    f(i, 0) += i;
    f(i, 1) += i;
    f(i, 2) -= i;

    t_vec3  v = f.get<t_vec3>(i);
    f.getv(i, v);
    v = f.getv(i);
}

__global__ void test_ker_field2(t_field2 f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    f(i, 0, 0) += i;
    f(i, 0, 1) += i;
    f(i, 0, 2) -= i;
    f(i, 1, 0) += i;
    f(i, 1, 1) += i;
    f(i, 1, 2) -= i;

    t_vec3  v = f.get<t_vec3>(i,0);
    f.getv(i, 0, v);
    v = f.getv(i, 0);
}

__global__ void test_ker_field3(t_field3 f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2) {
        f(i, i1, i2, 0) += i*(i1+1)+i2;
        f(i, i1, i2, 1) += i*(i1+1)+i2;
        f(i, i1, i2, 2) -= i*(i1+1)+i2;
    }

    t_vec3  v = f.get<t_vec3>(i,0,0);
    f.getv(i, 0, 0, v);
    v = f.getv(i, 0, 0);
}

__global__ void test_ker_field4(t_field4 f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i1 = 0;i1 < 2;++i1)
    for (int i2 = 0;i2 < 4;++i2)
    for (int i3 = 0;i3 < 5;++i3) {
        f(i, i1, i2, i3, 0) += i*(i1+1)+(i2*5)/(i3+1);
        f(i, i1, i2, i3, 1) += i*(i1+1)+(i2*5)/(i3+1);
        f(i, i1, i2, i3, 2) -= i*(i1+1)+(i2*5)/(i3+1);
    }

    t_vec3  v = f.get<t_vec3>(i,0,0,0);
    f.getv(i, 0, 0, 0, v);
    v = f.getv(i, 0, 0, 0);
}*/

bool    test_field0()
{
    t_field0        f;
    f.init(100, 0);

    t_field0_view   view(f, false);
    for (int i = 0;i < 100;++i) {
        view(i) = 1;
    }
    view.release();

    //call some kernel
    /*dim3 dimBlock(100,1);
    dim3 dimGrid(1,1);
    test_ker_field0<<<dimGrid, dimBlock>>>(f);*/

    bool    result = true;

    t_field0_view   view2(f, true);
    for (int i = 0;i < 100;++i) {
        if (view2(i) != 1+i) {
            printf("test_field1_ndim2: i = %d: %f != %f \n", i, view2(i), 1+i);
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f\n", i, view2(i));
#endif
    }
    view2.release();

    return result;
}

bool    test_field1()
{
    t_field1        f;
    f.init(100);

    t_field1_view   view(f, false);
    for (int i = 0;i < 100;++i) {
        view(i, 0) = 1;
        view(i, 1) = i;
        view(i, 2) = i;
    }
    view.release();

    //call some kernel
    /*dim3 dimBlock(100,1);
    dim3 dimGrid(1,1);
    test_ker_field1<<<dimGrid, dimBlock>>>(f);*/
    
    bool    result = true;

    t_field1_view   view2(f, true);
    for (int i = 0;i < 100;++i) {
        if (view2(i, 0) != 1+i) {
            printf("test_field1_ndim2: i = %d: %f != %f \n", i, view2(i, 0), float(1+i));
            result = false;
        }
        if (view2(i, 1) != i+i) {
            printf("test_field1_ndim2: i = %d: %f != %f \n", i, view2(i, 1), float(i+i));
            result = false;
        }
        if (view2(i, 2) != i-i) {
            printf("test_field1_ndim2: i = %d: %f != %f \n", i, view2(i, 2), float(i-i));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f, %f, %f\n", i, view2(i, 0), view2(i, 1), view2(i, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_field2()
{
    t_field2        f;
    f.init(100);

    t_field2_view   view(f, false);
    for (int i = 0;i < 100;++i) {
        view(i, 0, 0) = 1;
        view(i, 0, 1) = i;
        view(i, 0, 2) = i;

        view(i, 1, 0) = 1+2;
        view(i, 1, 1) = i+2;
        view(i, 1, 2) = i+2;
    }
    view.release();

    //call some kernel
    /*dim3 dimBlock(100,1);
    dim3 dimGrid(1,1);
    test_ker_field2<<<dimGrid, dimBlock>>>(f);*/

    bool    result = true;

    t_field2_view   view2(f, true);
    for (int i = 0;i < 100;++i) {
        if (view2(i, 0, 0) != 1+i) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 0, 0), float(1+i));
            result = false;
        }
        if (view2(i, 0, 1) != i+i) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 0, 1), float(i+i));
            result = false;
        }
        if (view2(i, 0, 2) != i-i) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 0, 2), float(i-i));
            result = false;
        }
        if (view2(i, 1, 0) != 1+i+2) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 1, 0), float(1+i+2));
            result = false;
        }
        if (view2(i, 1, 1) != i+i+2) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 1, 1), float(i+i+2));
            result = false;
        }
        if (view2(i, 1, 2) != i-i+2) {
            printf("test_field2_ndim2: i = %d: %f != %f \n", i, view2(i, 1, 2), float(i-i+2));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f, %f, %f, %f, %f, %f\n", i, view2(i, 0, 0), view2(i, 0, 1), view2(i, 0, 2),  view2(i, 1, 0), view2(i, 1, 1), view2(i, 1, 2));
#endif
    }
    view2.release();

    return result;
}

bool    test_field3()
{
    t_field3        f;
    f.init(100);

    t_field3_view   view(f, false);
    for (int i = 0;i < 100;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2) {
            view(i, i1, i2, 0) = 1*(i2+1)+i1;
            view(i, i1, i2, 1) = i*(i2+1)+i1;
            view(i, i1, i2, 2) = i*(i2+1)+i1;
        }
    }
    view.release();

    //call some kernel
    /*dim3 dimBlock(100,1);
    dim3 dimGrid(1,1);
    test_ker_field3<<<dimGrid, dimBlock>>>(f);*/

    bool    result = true;

    t_field3_view   view2(f, true);
    for (int i = 0;i < 100;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2) {
            if (view2(i, i1, i2, 0) != 1*(i2+1)+i1+(i*(i1+1)+i2)) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, 0), float(1*(i2+1)+i1+(i*(i1+1)+i2)));
                result = false;
            }
            if (view2(i, i1, i2, 1) != i*(i2+1)+i1+(i*(i1+1)+i2)) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, 1), float(i*(i2+1)+i1+(i*(i1+1)+i2)));
                result = false;
            }
            if (view2(i, i1, i2, 2) != i*(i2+1)+i1-(i*(i1+1)+i2)) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, 2), float(i*(i2+1)+i1-(i*(i1+1)+i2)));
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

bool    test_field4()
{
    t_field4        f;
    f.init(100);

    t_field4_view   view(f, false);
    for (int i = 0;i < 100;++i) {
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
    /*dim3 dimBlock(100,1);
    dim3 dimGrid(1,1);
    test_ker_field4<<<dimGrid, dimBlock>>>(f);*/

    bool    result = true;

    t_field4_view   view2(f, true);
    for (int i = 0;i < 100;++i) {
        for (int i1 = 0;i1 < 2;++i1)
        for (int i2 = 0;i2 < 4;++i2)
        for (int i3 = 0;i3 < 5;++i3) {
            if (view2(i, i1, i2, i3, 0) != 1*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, i3, 0), float(1*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))));
                result = false;
            }
            if (view2(i, i1, i2, i3, 1) != i*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, i3, 1), float(i*(i2+1)+i1+i3+(i*(i1+1)+(i2*5)/(i3+1))));
                result = false;
            }
            if (view2(i, i1, i2, i3, 2) != i*(i2+1)+i1+i3*2-(i*(i1+1)+(i2*5)/(i3+1))) {
                printf("test_field3_ndim2: i = %d: %f != %f \n", i, view2(i, i1, i2, i3, 2), float(i*(i2+1)+i1+i3*2-(i*(i1+1)+(i2*5)/(i3+1))));
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
    t_field0        f0,f0_;
    t_field1        f1,f1_;
    t_field2        f2,f2_;
    t_field2        f3,f3_;
    t_field2        f4,f4_;
    f0.init(100);
    f1.init(100);
    f2.init(100);
    f3.init(100);
    f4.init(100);
    f0_ = f0;
    f1_ = f1;
    f2_ = f2;
    f3_ = f3;
    f4_ = f4;

    return true;
}

int main()
{
    try {

    int err_code = 0;

    if (test_field0()) {
        printf("test_field0 seems to be OK\n");
    } else {
        printf("test_field0 failed\n");
        err_code = 2;
    }

    if (test_field1()) {
        printf("test_field1 seems to be OK\n");
    } else {
        printf("test_field1 failed\n");
        err_code = 2;
    }
    
    if (test_field2()) {
        printf("test_field2 seems to be OK\n");
    } else {
        printf("test_field2 failed\n");
        err_code = 2;
    }
    
    if (test_field3()) {
        printf("test_field3 seems to be OK\n");
    } else {
        printf("test_field3 failed\n");
        err_code = 2;
    }
    
    if (test_field4()) {
        printf("test_field4 seems to be OK\n");
    } else {
        printf("test_field4 failed\n");
        err_code = 2;
    }
    
    if (test_assign_operator()) {
        printf("test_assign_operator seems to be OK\n");
    } else {
        printf("test_assign_operator failed\n");
        err_code = 2;
    }

    return err_code;

    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }
}