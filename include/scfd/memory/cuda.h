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

#ifndef __SCFD_MEMORY_CUDA_H__
#define __SCFD_MEMORY_CUDA_H__

#include <stdexcept>
#include <cuda_runtime.h>
#include <scfd/utils/cuda_safe_call.h>

namespace scfd
{
namespace memory
{

struct cuda_host;

struct cuda_device
{
    typedef     cuda_host       host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;

    static const bool           is_host_visible = false;
    static const bool           prefer_array_of_structs = false;

    /// NOTE: cudaMalloc returns NULL for size==0 without error,
    /// however it's not stated explicitly in documentation
    static void    malloc(pointer_type* p, size_t size)
    {
        CUDA_SAFE_CALL(cudaMalloc(p, size));
    }
    /// NOTE: cudaFree returns no error when called with NULL,
    /// however it's not stated explicitly in documentation
    static void    free(pointer_type p)
    {
        CUDA_SAFE_CALL(cudaFree(p));
    }

    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
    }
};

struct cuda_host
{
    typedef     cuda_host       host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;
    static const bool           is_host_visible = true;
    static const bool           prefer_array_of_structs = false;

    /// NOTE: cudaMallocHost DOES NOT returns NULL for size==0 
    /// (it does not change ptr without generating error),
    /// and this behaviour is not stated explicitly in documentation
    static void    malloc(pointer_type* p, size_t size)
    {
        if (size != 0)
        {
            CUDA_SAFE_CALL(cudaMallocHost(p, size,cudaHostAllocDefault));
        }
        else
        {
            *p = NULL;
        }
    }
    /// NOTE: cudaFreeHost returns no error when called with NULL,
    /// however it's not stated explicitly in documentation
    static void    free(pointer_type p)
    {
        CUDA_SAFE_CALL(cudaFreeHost(p));
    }

    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToHost) );
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        CUDA_SAFE_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToHost) );
    }
};

}

}

#endif
