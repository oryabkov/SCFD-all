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
using current = serial_cpu<PLATFORM_ORDINAL>;
}
}

#elif defined( PLATFORM_OMP )
#    include "omp.h"
namespace scfd
{
namespace backend
{
using current = omp<PLATFORM_ORDINAL>;
}
}

#elif defined( PLATFORM_CUDA )
#    include "cuda.h"
namespace scfd
{
namespace backend
{
using current = cuda<PLATFORM_ORDINAL>;
}
}

#elif defined( PLATFORM_HIP )
#    include "hip.h"
namespace scfd
{
namespace backend
{
using current = hip<PLATFORM_ORDINAL>;
}
}

#elif defined( PLATFORM_SYCL )
#    include "sycl.h"
namespace scfd
{
namespace backend
{
using current = sycl<PLATFORM_ORDINAL>;
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
using memory             = current::memory_type;
using timer_event        = current::timer_event_type;


using for_each = typename current::for_each_type;
template <int Dim>
using for_each_nd      = typename current::template for_each_nd_type<Dim>;
using reduce           = typename current::reduce_type;
using sort             = typename current::sort_type;
using unique           = typename current::unique_type;
using exclusive_scan   = typename current::exclusive_scan_type;
using copy             = typename current::copy_type;
using inclusive_scan   = typename current::inclusive_scan_type;
using sort_by_key      = typename current::sort_by_key_type;
using reduce_by_key    = typename current::reduce_by_key_type;
using set_intersection = typename current::set_intersection_type;
using sequence         = typename current::sequence_type;
using count_by_key     = typename current::count_by_key_type;
}
}

#endif
