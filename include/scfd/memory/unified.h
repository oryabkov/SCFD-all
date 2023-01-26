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

#ifndef __SCFD_MEMORY_UNIFIED_H__
#define __SCFD_MEMORY_UNIFIED_H__

#include <stdexcept>
#include <cuda_runtime.h>
#include <scfd/utils/cuda_safe_call.h>

namespace scfd
{
namespace memory
{

struct unified
{
    typedef     unified         host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;

    static const bool           is_host_visible = true;
    static const bool           prefer_array_of_structs = false;

    // NOTE: flags specifies the default stream association for this allocation. 
    // flags must be one of cudaMemAttachGlobal or cudaMemAttachHost. 
    // The default value for flags is cudaMemAttachGlobal. 
    // If cudaMemAttachGlobal is specified, then this memory is accessible from any stream on any device. 
    // If cudaMemAttachHost is specified, then the allocation should not be accessed from devices that have a zero value for the device attribute cudaDevAttrConcurrentManagedAccess; 
    // an explicit call to cudaStreamAttachMemAsync will be required to enable access on such devices. 

    /// NOTE: cudaMalloc returns NULL for size==0 without error,
    /// however it's not stated explicitly in documentation
    static void    malloc(pointer_type* p, size_t size, unsigned int flags = cudaMemAttachGlobal)
    {
        CUDA_SAFE_CALL(cudaMallocManaged(p, size, flags));
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


}

}

#endif
