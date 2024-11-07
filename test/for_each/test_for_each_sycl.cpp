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

bool    test_field0()
{
    for_each_t for_each;
    auto device_usm = sycl::malloc_device<int>(SZ_X, sycl_device_queue);
    auto host_usm   = sycl::malloc_host  <int>(SZ_X, sycl_device_queue);

    for_each([=](int idx){ device_usm[idx] = 1 - idx*idx; }, 0, SZ_X);

    sycl_device_queue.copy(device_usm, host_usm, SZ_X).wait();

    bool    result = true;
    for (int i = 0;i < SZ_X;++i)
    {
        if (host_usm[i] != 1 - i*i)
        {
            printf("test_field0: i = %d: %d != %d \n", i, host_usm[i], 1 - i*i);
            result = false;
        }
        #ifdef DO_RESULTS_OUTPUT
        printf("%d, %d, %d\n", i, host_usm[i]);
        #endif
    }

    sycl::free(device_usm, sycl_device_queue);
    sycl::free(host_usm,   sycl_device_queue);
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
