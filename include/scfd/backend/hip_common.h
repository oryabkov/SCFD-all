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

#ifndef __SCFD_BACKEND_HIP_COMMON_H__
#define __SCFD_BACKEND_HIP_COMMON_H__

#include <hip/hip_runtime.h>
#include <scfd/backend/common.h>
#include <scfd/copy/hip.h>
#include <scfd/count_by_key/thrust.h>
#include <scfd/exclusive_scan/thrust.h>
#include <scfd/for_each/hip_impl.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/inclusive_scan/thrust.h>
#include <scfd/memory/hip.h>
#include <scfd/platform/config.h>
#include <scfd/reduce/thrust.h>
#include <scfd/reduce_by_key/thrust.h>
#include <scfd/sequence/thrust.h>
#include <scfd/set_intersection/thrust.h>
#include <scfd/sort/thrust.h>
#include <scfd/sort_by_key/thrust.h>
#include <scfd/unique/thrust.h>
#include <scfd/utils/hip_safe_call.h>
#include <scfd/utils/hip_timer_event.h>

namespace scfd
{
namespace backend
{

template <class Ordinal = PLATFORM_ORDINAL>
struct hip_common
{
    using ordinal_type            = Ordinal;
    using memory_type             = scfd::memory::hip_device;
    using device_memory_info_type = scfd::backend::detail::device_memory_info;
    using host_memory_info_type   = scfd::backend::detail::host_memory_info;
    using timer_event_type        = scfd::utils::hip_timer_event;
    using for_each_type           = scfd::for_each::hip<Ordinal>;
    template <int Dim>
    using for_each_nd_type      = scfd::for_each::hip_nd<Dim, Ordinal>;
    using reduce_type           = scfd::reduce::thrust<Ordinal>;
    using sort_type             = scfd::sort::thrust<Ordinal>;
    using unique_type           = scfd::unique::thrust<Ordinal>;
    using exclusive_scan_type   = scfd::exclusive_scan::thrust<Ordinal>;
    using copy_type             = scfd::copy::hip<Ordinal>;
    using inclusive_scan_type   = scfd::inclusive_scan::thrust<Ordinal>;
    using sort_by_key_type      = scfd::sort_by_key::thrust<Ordinal>;
    using reduce_by_key_type    = scfd::reduce_by_key::thrust<Ordinal>;
    using set_intersection_type = scfd::set_intersection::thrust<Ordinal>;
    using sequence_type         = scfd::sequence::thrust<Ordinal>;
    using count_by_key_type     = scfd::count_by_key::thrust<Ordinal>;

    static const char *name()
    {
        return "hip";
    }

    static void synchronize()
    {
        SCFD_HIP_SAFE_CALL( hipDeviceSynchronize() );
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info_type get_device_memory_info()
    {
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        SCFD_HIP_SAFE_CALL( hipMemGetInfo( &free_bytes, &total_bytes ) );
        return device_memory_info_type( free_bytes, total_bytes, true, true );
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

#endif
