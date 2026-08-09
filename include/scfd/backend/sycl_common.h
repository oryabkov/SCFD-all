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

#ifndef __SCFD_BACKEND_SYCL_COMMON_H__
#define __SCFD_BACKEND_SYCL_COMMON_H__

#include <cstddef>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <scfd/backend/common.h>
#include <scfd/copy/sycl_copy_impl.h>
#include <scfd/count_by_key/sycl_count_by_key_impl.h>
#include <scfd/exclusive_scan/sycl_exclusive_scan_impl.h>
#include <scfd/for_each/sycl_impl.h>
#include <scfd/for_each/sycl_nd_impl.h>
#include <scfd/inclusive_scan/sycl_inclusive_scan_impl.h>
#include <scfd/memory/sycl.h>
#include <scfd/reduce/sycl_reduce_impl.h>
#include <scfd/reduce_by_key/sycl_reduce_by_key_impl.h>
#include <scfd/sequence/sycl_sequence_impl.h>
#include <scfd/set_intersection/sycl_set_intersection_impl.h>
#include <scfd/sort/sycl_sort_impl.h>
#include <scfd/sort_by_key/sycl_sort_by_key_impl.h>
#include <scfd/unique/sycl_unique_impl.h>
#include <scfd/utils/init_sycl.h>
#include <scfd/utils/system_timer_event.h>

#define MAKE_SYCL_DEVICE_COPYABLE( kernel )                                                                            \
    template <>                                                                                                        \
    struct sycl::is_device_copyable<typename kernel> : std::true_type                                                  \
    {                                                                                                                  \
    }

namespace scfd
{
namespace backend
{

struct sycl_common
{
    using memory_type             = scfd::memory::sycl_device;
    using device_memory_info_type = scfd::backend::detail::device_memory_info;
    using host_memory_info_type   = scfd::backend::detail::host_memory_info;
    using timer_event_type        = scfd::utils::system_timer_event;
    template <class Ordinal = int>
    using for_each_type = scfd::for_each::sycl_<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type      = scfd::for_each::sycl_nd<Dim, Ordinal>;
    using reduce_type           = scfd::sycl_reduce<>;
    using sort_type             = scfd::sycl_sort<>;
    using unique_type           = scfd::sycl_unique<>;
    using exclusive_scan_type   = scfd::sycl_exclusive_scan<>;
    using copy_type             = scfd::sycl_copy<>;
    using inclusive_scan_type   = scfd::sycl_inclusive_scan<>;
    using sort_by_key_type      = scfd::sycl_sort_by_key<>;
    using reduce_by_key_type    = scfd::sycl_reduce_by_key<>;
    using set_intersection_type = scfd::sycl_set_intersection<>;
    using sequence_type         = scfd::sycl_sequence<>;
    using count_by_key_type     = scfd::sycl_count_by_key<>;

    static const char *name()
    {
        return "sycl";
    }

    static void synchronize()
    {
        sycl_device_queue.wait_and_throw();
    }

    static void device_synchronize()
    {
        synchronize();
    }

    static device_memory_info_type get_device_memory_info()
    {
        const std::size_t total_bytes =
            static_cast<std::size_t>( sycl_device_queue.get_device().get_info<::sycl::info::device::global_mem_size>()
            );
        return device_memory_info_type( 0, total_bytes, false, true );
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

#endif
