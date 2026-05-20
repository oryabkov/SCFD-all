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

#ifndef __SCFD_BACKEND_RUNTIME_HIP_H__
#define __SCFD_BACKEND_RUNTIME_HIP_H__

#include <hip/hip_runtime.h>
#include <scfd/backend/runtime/common.h>
#include <scfd/utils/hip_safe_call.h>
#include <scfd/utils/init_hip.h>
#include <scfd/utils/hip_timer_event.h>

namespace scfd
{
namespace backend
{
namespace detail
{

struct hip_runtime
{
    using timer_event_type = scfd::utils::hip_timer_event;

    template <class Log>
    static int init_device( Log &log, int device_id = 0 )
    {
        return scfd::utils::init_hip( log, -2, device_id );
    }

    static int init_device( int device_id = 0 )
    {
        return scfd::utils::init_hip( -2, device_id );
    }

    template <bool WrapProcsDevices = false, class Log, class Comm>
    static int init_device_mpi( Log &log, const Comm &comm, int shift_index = 0 );

    template <bool WrapProcsDevices = false, class Comm>
    static int init_device_mpi( const Comm &comm, int shift_index = 0 );

    static void synchronize()
    {
        HIP_SAFE_CALL( hipDeviceSynchronize() );
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info get_memory_info()
    {
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        HIP_SAFE_CALL( hipMemGetInfo( &free_bytes, &total_bytes ) );
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
