// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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
//

#ifndef __SCFD_BACKEND_SYCL_H__
#define __SCFD_BACKEND_SYCL_H__

#include <scfd/memory/sycl.h>
#include <scfd/for_each/sycl_impl.h>
#include <scfd/for_each/sycl_nd_impl.h>
#include <scfd/backend/reduce/sycl_reduce_impl.h>
#include <scfd/backend/sort/sycl_sort_impl.h>
#include <scfd/backend/unique/sycl_unique_impl.h>
#include <scfd/backend/exclusive_scan/sycl_exclusive_scan_impl.h>
#include <scfd/backend/copy/sycl_copy_impl.h>
#include <scfd/backend/inclusive_scan/sycl_inclusive_scan_impl.h>
#include <scfd/backend/sort_by_key/sycl_sort_by_key_impl.h>
#include <scfd/backend/reduce_by_key/sycl_reduce_by_key_impl.h>
#include <scfd/backend/set_intersection/sycl_set_intersection_impl.h>
#include <scfd/backend/sequence/sycl_sequence_impl.h>
#include <scfd/backend/runtime/sycl.h>

#define MAKE_SYCL_DEVICE_COPYABLE( kernel )                                                                            \
    template <>                                                                                                        \
    struct sycl::is_device_copyable<typename kernel> : std::true_type                                                  \
    {                                                                                                                  \
    }

namespace scfd
{
namespace backend
{
struct sycl
{
    using memory_type = scfd::memory::sycl_device;
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
    using runtime_type          = scfd::backend::detail::sycl_runtime;
};
}
}

#endif // __SCFD_BACKEND_SYCL_H__
