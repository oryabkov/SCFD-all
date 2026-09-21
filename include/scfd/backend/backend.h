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

#ifndef __SCFD_BACKEND_H__
#define __SCFD_BACKEND_H__

#if ( defined( PLATFORM_SERIAL_CPU ) + defined( PLATFORM_OMP ) + defined( PLATFORM_CUDA ) + defined( PLATFORM_HIP ) +  \
      defined( PLATFORM_SYCL ) ) != 1
#    error "Select exactly one execution platform for backend"
#endif

#if defined( PLATFORM_SERIAL_CPU )
#    include "serial_cpu.h"
namespace scfd
{
namespace backend
{
template <class Ordinal = PLATFORM_ORDINAL>
using current = serial_cpu<Ordinal>;
}
}

#elif defined( PLATFORM_OMP )
#    include "omp.h"
namespace scfd
{
namespace backend
{
template <class Ordinal = PLATFORM_ORDINAL>
using current = omp<Ordinal>;
}
}

#elif defined( PLATFORM_CUDA )
#    include "cuda.h"
namespace scfd
{
namespace backend
{
template <class Ordinal = PLATFORM_ORDINAL>
using current = cuda<Ordinal>;
}
}

#elif defined( PLATFORM_HIP )
#    include "hip.h"
namespace scfd
{
namespace backend
{
template <class Ordinal = PLATFORM_ORDINAL>
using current = hip<Ordinal>;
}
}

#elif defined( PLATFORM_SYCL )
#    include "sycl.h"
namespace scfd
{
namespace backend
{
template <class Ordinal = PLATFORM_ORDINAL>
using current = sycl<Ordinal>;
}
}

#endif

namespace scfd
{
namespace backend
{
// Useful aliases.
using device_memory_info = detail::device_memory_info;
using host_memory_info   = detail::host_memory_info;
using memory             = current<>::memory_type;
using timer_event        = current<>::timer_event_type;

template <class Ordinal = PLATFORM_ORDINAL>
using for_each = typename current<Ordinal>::for_each_type;
template <int Dim, class Ordinal = PLATFORM_ORDINAL>
using for_each_nd = typename current<Ordinal>::template for_each_nd_type<Dim>;
template <class Ordinal = PLATFORM_ORDINAL>
using reduce = typename current<Ordinal>::reduce_type;
template <class Ordinal = PLATFORM_ORDINAL>
using sort = typename current<Ordinal>::sort_type;
template <class Ordinal = PLATFORM_ORDINAL>
using unique = typename current<Ordinal>::unique_type;
template <class Ordinal = PLATFORM_ORDINAL>
using exclusive_scan = typename current<Ordinal>::exclusive_scan_type;
template <class Ordinal = PLATFORM_ORDINAL>
using copy = typename current<Ordinal>::copy_type;
template <class Ordinal = PLATFORM_ORDINAL>
using inclusive_scan = typename current<Ordinal>::inclusive_scan_type;
template <class Ordinal = PLATFORM_ORDINAL>
using sort_by_key = typename current<Ordinal>::sort_by_key_type;
template <class Ordinal = PLATFORM_ORDINAL>
using reduce_by_key = typename current<Ordinal>::reduce_by_key_type;
template <class Ordinal = PLATFORM_ORDINAL>
using set_intersection = typename current<Ordinal>::set_intersection_type;
template <class Ordinal = PLATFORM_ORDINAL>
using sequence = typename current<Ordinal>::sequence_type;
template <class Ordinal = PLATFORM_ORDINAL>
using count_by_key = typename current<Ordinal>::count_by_key_type;
template <class Ordinal = PLATFORM_ORDINAL>
using runtime = current<Ordinal>;
}
}

#endif
