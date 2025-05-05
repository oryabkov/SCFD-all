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

#ifndef __SCFD_MEMORY_HIP_H__
#define __SCFD_MEMORY_HIP_H__

#include <stdexcept>
#include <hip/hip_runtime.h>
#include <scfd/utils/hip_safe_call.h>

namespace scfd
{
namespace memory
{

struct hip_host;

struct hip_device
{
    typedef     hip_host        host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;

    static const bool           is_host_visible = false;
    static const bool           prefer_array_of_structs = false;

    /// NOTE: hipMalloc returns NULL for size==0 without error,
    /// however it's not stated explicitly in documentation
    static void    malloc(pointer_type* p, size_t size)
    {
        HIP_SAFE_CALL(hipMalloc(p, size));
    }
    /// NOTE: hipFree returns no error when called with NULL,
    /// however it's not stated explicitly in documentation
    static void    free(pointer_type p)
    {
        HIP_SAFE_CALL(hipFree(p));
    }

    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
    }
    static void    copy(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
    }
};

struct hip_host
{
    typedef     hip_host        host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;
    static const bool           is_host_visible = true;
    static const bool           prefer_array_of_structs = false;

    /// NOTE: hipMallocHost DOES NOT returns NULL for size==0
    /// (it does not change ptr without generating error),
    /// and this behaviour is not stated explicitly in documentation
    static void    malloc(pointer_type* p, size_t size)
    {
        if (size != 0)
        {
            HIP_SAFE_CALL(hipHostMalloc(p, size,0));
        }
        else
        {
            *p = NULL;
        }
    }
    /// NOTE: hipFreeHost returns no error when called with NULL,
    /// however it's not stated explicitly in documentation
    static void    free(pointer_type p)
    {
        HIP_SAFE_CALL(hipHostFree(p));
    }

    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToHost) );
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToHost) );
    }
    static void    copy(size_t size, const_pointer_type src, pointer_type dst)
    {
        HIP_SAFE_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToHost) );
    }
};

}

}

// #if 0
#include "thrust_ptr_hip.h" // TODO: add thrust support for hip
// #endif

#endif
