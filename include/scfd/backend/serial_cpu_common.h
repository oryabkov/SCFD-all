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

#ifndef __SCFD_BACKEND_SERIAL_CPU_COMMON_H__
#define __SCFD_BACKEND_SERIAL_CPU_COMMON_H__

#include <scfd/backend/common.h>
#include <scfd/copy/serial.h>
#include <scfd/count_by_key/serial.h>
#include <scfd/exclusive_scan/serial.h>
#include <scfd/for_each/serial_cpu.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/inclusive_scan/serial.h>
#include <scfd/memory/host.h>
#include <scfd/reduce/serial.h>
#include <scfd/reduce_by_key/serial.h>
#include <scfd/sequence/serial.h>
#include <scfd/set_intersection/serial.h>
#include <scfd/sort/serial.h>
#include <scfd/sort_by_key/serial.h>
#include <scfd/unique/serial.h>
#include <scfd/utils/system_timer_event.h>

namespace scfd
{
namespace backend
{

struct serial_cpu_common
{
    using memory_type             = scfd::memory::host;
    using device_memory_info_type = scfd::backend::detail::device_memory_info;
    using host_memory_info_type   = scfd::backend::detail::host_memory_info;
    using timer_event_type        = scfd::utils::system_timer_event;
    template <class Ordinal = int>
    using for_each_type = scfd::for_each::serial_cpu<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type      = scfd::for_each::serial_cpu_nd<Dim, Ordinal>;
    using reduce_type           = scfd::reduce::serial<>;
    using sort_type             = scfd::sort::serial<>;
    using unique_type           = scfd::unique::serial<>;
    using exclusive_scan_type   = scfd::exclusive_scan::serial<>;
    using copy_type             = scfd::copy::serial<>;
    using inclusive_scan_type   = scfd::inclusive_scan::serial<>;
    using sort_by_key_type      = scfd::sort_by_key::serial<>;
    using reduce_by_key_type    = scfd::reduce_by_key::serial<>;
    using set_intersection_type = scfd::set_intersection::serial<>;
    using sequence_type         = scfd::sequence::serial<>;
    using count_by_key_type     = scfd::count_by_key::serial<>;

    static const char *name()
    {
        return "serial_cpu";
    }

    static void synchronize()
    {
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info_type get_device_memory_info()
    {
        return device_memory_info_type();
    }

    static device_memory_info_type get_memory_info()
    {
        return get_device_memory_info();
    }

    static device_memory_info_type memory_info()
    {
        return get_memory_info();
    }

    static host_memory_info_type get_host_memory_info()
    {
        return scfd::backend::detail::get_host_memory_info();
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

#endif
