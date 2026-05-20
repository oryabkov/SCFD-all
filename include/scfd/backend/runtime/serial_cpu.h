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

#ifndef __SCFD_BACKEND_RUNTIME_SERIAL_CPU_H__
#define __SCFD_BACKEND_RUNTIME_SERIAL_CPU_H__

#include <scfd/backend/runtime/common.h>
#include <scfd/utils/system_timer_event.h>

namespace scfd
{
namespace backend
{
namespace detail
{

struct serial_cpu_runtime
{
    using timer_event_type = scfd::utils::system_timer_event;

    template <class Log>
    static int init_device( Log &, int = -1 )
    {
        return -1;
    }

    static int init_device( int = -1 )
    {
        return -1;
    }

    template <bool WrapProcsDevices = false, class Log, class Comm>
    static int init_device_mpi( Log &, const Comm &, int = 0 )
    {
        return -1;
    }

    template <bool WrapProcsDevices = false, class Comm>
    static int init_device_mpi( const Comm &, int = 0 )
    {
        return -1;
    }

    static void synchronize()
    {
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info get_memory_info()
    {
        return device_memory_info();
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
        return false;
    }

    static bool is_device_backend()
    {
        return false;
    }

    static bool reports_free_memory()
    {
        return false;
    }

    static bool reports_total_memory()
    {
        return false;
    }
};

}
}
}

#endif
