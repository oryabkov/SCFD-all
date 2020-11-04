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
#include <scfd/arrays/tensorN_array_visible.h>
#include <scfd/arrays/tensorN_array_nd_visible.h>

//#define DO_RESULTS_OUTPUT

typedef scfd::memory::cuda_device                                     mem_t;
typedef int                                                           t_idx;
typedef scfd::static_vec::vec<float,3>                                t_vec3;
typedef scfd::arrays::tensor0_array_visible<float,mem_t>              array0_vis_t;
typedef array0_vis_t::array_type                                      array0_t;
typedef scfd::arrays::tensor1_array_visible<float,mem_t,3>            array1_vis_t;
typedef scfd::arrays::tensor2_array_visible<float,mem_t,2,3>          array2_vis_t;
typedef scfd::arrays::tensor3_array_visible<float,mem_t,2,4,3>        array3_vis_t;
typedef scfd::arrays::tensor4_array_visible<float,mem_t,2,4,5,3>      array4_vis_t;

#define NDIM                                                    2
typedef scfd::static_vec::vec<int,NDIM>                               idx_nd_t;
typedef scfd::arrays::tensor1_array_nd_visible<float,NDIM,mem_t,3>    array1_nd_vis_t;
typedef array1_nd_vis_t::array_type                                   array1_nd_t;
typedef scfd::arrays::tensor1_array_nd_visible<float,NDIM,mem_t,0>    array1_nd_vis_dyn_t;

int     sz1 = 100, sz2 = 100;
int     block_sz1 = 256;
int     block2_sz1 = 16,block2_sz2 = 16;

__global__ void test_ker_array0(array0_t f)
{
    int     i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < f.get_dim<0>())) return;
    f(i) += i;
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

bool    test_array0()
{
    array0_vis_t        f;
    dim3                dimBlock(block_sz1,1);
    dim3                dimGrid((sz1/block_sz1)+1,1);
    f.init(sz1, 0);

    for (int i = 0;i < sz1;++i) {
        f(i) = 1;
    }
    f.sync_to_array();

    //test correctness
    test_ker_array0<<<dimGrid, dimBlock>>>(f.array());

    bool            result = true;

    f.sync_from_array();
    for (int i = 0;i < sz1;++i) {
        if (f(i) != 1+i) {
            printf("test_array0: i = %d: %f != %f \n", i, f(i), float(1+i));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %f\n", i, f(i));
#endif
    }

    return result;
}

bool    test_array1_nd_1(int shift1, int shift2)
{
    array1_nd_vis_t f;
    dim3            dimBlock(block2_sz1,block2_sz2);
    dim3            dimGrid((sz1/block2_sz1)+1,(sz2/block2_sz2)+1);
    if ((shift1 == 0)&&(shift2 == 0))
        f.init(idx_nd_t(sz1,sz2));
    else
        f.init(idx_nd_t(sz1,sz2), idx_nd_t(shift1,shift2));

    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        f(idx_nd_t(i,j), 0) = i;
        f(idx_nd_t(i,j), 1) = i+j;
        f(idx_nd_t(i,j), 2) = i*2+j;
    }
    f.sync_to_array();

    //call some kernel
    test_ker_array1_nd_1<<<dimGrid, dimBlock>>>(f.array());

    bool    result = true;

    f.sync_from_array();
    for (int i = shift1;i < sz1+shift1;++i)
    for (int j = shift2;j < sz2+shift2;++j) {
        if (f(idx_nd_t(i,j), 0) != i+1) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 0: %f != %f \n", i, j, f(idx_nd_t(i,j), 0), float(i+1));
            result = false;
        }
        if (f(idx_nd_t(i,j), 1) != i+j-i) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 1: %f != %f \n", i, j, f(idx_nd_t(i,j), 1), float(i+j-i));
            result = false;
        }
        if (f(idx_nd_t(i,j), 2) != i*2+j-j) {
            printf("test_array1_nd_1: i = %d, j = %d, i0 = 2: %f != %f \n", i, j, f(idx_nd_t(i,j), 2), float(i*2+j-j));
            result = false;
        }
#ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %f, %f, %f\n", i, j, f(idx_nd_t(i,j), 0), f(idx_nd_t(i,j), 1), f(idx_nd_t(i,j), 2));
#endif
    }

    return result;
}

int main(int argc, char const *argv[])
{
    try {

    int err_code = 0;

    scfd::utils::init_cuda(-2,0);

    if (test_array0()) {
        printf("test_array0 seems to be OK\n");
    } else {
        printf("test_array0 failed\n");
        err_code = 2;
    }

    if (test_array1_nd_1(1,1)) {
        printf("test_array1_nd_1 seems to be OK\n");
    } else {
        printf("test_array1_nd_1 failed\n");
        err_code = 2;
    }

    return err_code;

    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }

}