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

#ifndef __SCFD_BACKEND_RUNTIME_SYCL_H__
#define __SCFD_BACKEND_RUNTIME_SYCL_H__

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <scfd/backend/runtime/common.h>
#include <scfd/utils/init_sycl.h>
#include <scfd/utils/system_timer_event.h>

namespace scfd
{
namespace backend
{
namespace detail
{

struct sycl_runtime
{
    using timer_event_type = scfd::utils::system_timer_event;

    template <class Log>
    static int init_device( Log &, int device_id = 0 )
    {
        return init_device( device_id );
    }

    static int init_device( int device_id = 0 )
    {
        std::vector<::sycl::device> devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
        if ( devices.empty() )
            throw std::runtime_error( "sycl_runtime::init_device: no visible SYCL GPU devices" );
        if ( device_id < 0 || device_id >= static_cast<int>( devices.size() ) )
            throw std::runtime_error(
                "sycl_runtime::init_device: requested device " + std::to_string( device_id ) +
                " is outside visible SYCL GPU device range"
            );
        sycl_device_queue = ::sycl::queue( devices[static_cast<std::size_t>( device_id )] );
        return device_id;
    }

    template <bool WrapProcsDevices = false, class Log, class Comm>
    static int init_device_mpi( Log &log, const Comm &comm, int shift_index = 0 );

    template <bool WrapProcsDevices = false, class Comm>
    static int init_device_mpi( const Comm &comm, int shift_index = 0 );

    static void synchronize()
    {
        sycl_device_queue.wait_and_throw();
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info get_memory_info()
    {
        const std::size_t total_bytes =
            static_cast<std::size_t>( sycl_device_queue.get_device().get_info<::sycl::info::device::global_mem_size>()
            );
        return device_memory_info( 0, total_bytes, false, true );
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
        return true;
    }

    static bool reports_free_memory()
    {
        return false;
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
