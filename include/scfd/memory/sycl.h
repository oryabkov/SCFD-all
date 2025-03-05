// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_MEMORY_SYCL_H__
#define __SCFD_MEMORY_SYCL_H__

#include <stdexcept>
#include <sycl/sycl.hpp>
#include <scfd/utils/init_sycl.h>

namespace scfd
{
namespace memory
{

struct sycl_host;

struct sycl_device
{
    typedef     sycl_host       host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;

    static const bool           is_host_visible = false;
    static const bool           prefer_array_of_structs = false;

    static void    malloc(pointer_type* p, size_t size)
    {
        *p = sycl::malloc_device<char>(size, sycl_device_queue);
    }
    static void    free(pointer_type p)
    {
        sycl::free(p, sycl_device_queue);
    }
    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
    static void    copy(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
};

struct sycl_host
{
    typedef     sycl_host       host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;
    
    static const bool           is_host_visible = true;
    static const bool           prefer_array_of_structs = true;

    static void    malloc(pointer_type* p, size_t size)
    {
        *p = sycl::malloc_host<char>(size, sycl_device_queue);
    }
    static void    free(pointer_type p)
    {
        sycl::free(p, sycl_device_queue);
    }
    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
    static void    copy(size_t size, const_pointer_type src, pointer_type dst)
    {
        sycl_device_queue.memcpy( dst, src, size ).wait();
    }
};

}

}

/// ISSUE is it ok? (alternative is to add some predefine in the following header or simply move this specialization here)
// #include "thrust_ptr_cuda.h"

#endif
