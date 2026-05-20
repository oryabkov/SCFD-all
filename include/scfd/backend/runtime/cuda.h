// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_BACKEND_RUNTIME_CUDA_H__
#define __SCFD_BACKEND_RUNTIME_CUDA_H__

#include <cuda_runtime.h>
#include <scfd/backend/runtime/common.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cuda_timer_event.h>

namespace scfd
{
namespace backend
{
namespace detail
{

struct cuda_runtime
{
    using timer_event_type = scfd::utils::cuda_timer_event;

    static void synchronize()
    {
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info get_memory_info()
    {
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        CUDA_SAFE_CALL( cudaMemGetInfo( &free_bytes, &total_bytes ) );
        return device_memory_info( free_bytes, total_bytes, true, true );
    }

    static device_memory_info memory_info()
    {
        return get_memory_info();
    }

    static device_memory_info get_device_memory_info()
    {
        return get_memory_info();
    }

    static bool uses_device_timer()
    {
        return true;
    }

    static bool is_device_backend()
    {
        return true;
    }

    static bool reports_free_memory()
    {
        return true;
    }

    static bool reports_total_memory()
    {
        return true;
    }
};

}
}
}

#endif
