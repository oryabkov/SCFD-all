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

#include <cstdio>
#include <stdexcept>

#include <scfd/static_vec/vec.h>
#include <scfd/utils/init_sycl.h>
#include <scfd/arrays/tensorN_array.h>

#include <scfd/memory/sycl.h>
#include <scfd/for_each/sycl.h>
#include <scfd/for_each/sycl_impl.h>

#define SZ_X    100

using for_each_t = scfd::for_each::sycl_impl<>;
using mem_t      = scfd::memory  ::sycl_device;

typedef scfd::arrays::tensor0_array<int,mem_t>                t_field0;
typedef scfd::arrays::tensor0_array_view<int,mem_t>           t_field0_view;
typedef scfd::arrays::tensor1_array<int,mem_t,3>              t_field1;
typedef scfd::arrays::tensor1_array_view<int,mem_t,3>         t_field1_view;
typedef scfd::arrays::tensor2_array<int,mem_t,3,4>            t_field2;
typedef scfd::arrays::tensor2_array_view<int,mem_t,3,4>       t_field2_view;

struct func_test_field0
{
    func_test_field0(const t_field0 &f_) : f(f_) {}
    mutable t_field0  f;
    
    void operator()(const int &idx) const
    {
        f(idx) += 1 - idx*idx;
    }
};

bool    test_field0()
{
    t_field0   f;
    f.init(SZ_X);

    t_field0_view     view;
    view.init(f, false);
    for (int i = 0;i < SZ_X;++i)
    {
        view(i) = i;
    }
    view.release();

    for_each_t             for_each;
    for_each(func_test_field0(f), 0, SZ_X);
    // do we need to wait?
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


int main()
{
    try {
    
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
/*
    if (test_field1())
    {
        printf("test_field1 seems to be OK\n");
    }
    else
    {
        printf("test_field1 failed\n");
        err_code = 2;
    }
*/
    return err_code;

    } catch(std::exception& e) {

    printf("exception caught: %s\n", e.what());
    return 1;

    }
}
