// Copyright © 2016-2022 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_TRASH_CUDA_MEMORY_H__
#define __SCFD_UTILS_TRASH_CUDA_MEMORY_H__

#include <cuda_runtime.h>˰
#include "cuda_safe_call.h"

namespace scfd
{
namespace utils
{

void trash_cuda_memory()
{
    size_t free_device_mem;
    size_t total_device_mem;
    CUDA_SAFE_CALL( cudaMemGetInfo(&free_device_mem, &total_device_mem) );
    void *all_ptr;
    size_t trash_device_mem = free_device_mem - 0x1000000;
    CUDA_SAFE_CALL( cudaMalloc(&all_ptr, trash_device_mem) );
    CUDA_SAFE_CALL( cudaMemset (all_ptr, 255, trash_device_mem ) );
    CUDA_SAFE_CALL( cudaFree(all_ptr) );
}

}

}

#endif
