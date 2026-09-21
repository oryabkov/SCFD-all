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

#ifndef __SCFD_PLATFORM_PLATFORM_H__
#define __SCFD_PLATFORM_PLATFORM_H__

#include <scfd/backend/backend.h>
#include <scfd/platform/config.h>

#ifdef PLATFORM_MPI
#    if defined( PLATFORM_SERIAL_CPU )
#        include "serial_cpu.h"
#    elif defined( PLATFORM_OMP )
#        include "omp.h"
#    elif defined( PLATFORM_CUDA )
#        include "cuda.h"
#    elif defined( PLATFORM_HIP )
#        include "hip.h"
#    elif defined( PLATFORM_SYCL )
#        include "sycl.h"
#    endif
#else
#    include "local.h"
#endif

namespace scfd
{
namespace platform
{

// PLATFORM_MPI is a presence flag. Without it, use a single-rank host
// communicator; backend selection remains independent of communication.
#ifdef PLATFORM_MPI
template <
    class Ordinal = PLATFORM_ORDINAL, class BigOrdinal = PLATFORM_BIG_ORDINAL,
    class Communicator = scfd::communication::mpi_comm_info>
#    if defined( PLATFORM_SERIAL_CPU )
using current = serial_cpu_mpi<Ordinal, BigOrdinal, Communicator>;
#    elif defined( PLATFORM_OMP )
using current = omp_mpi<Ordinal, BigOrdinal, Communicator>;
#    elif defined( PLATFORM_CUDA )
using current = cuda_mpi<Ordinal, BigOrdinal, Communicator>;
#    elif defined( PLATFORM_HIP )
using current = hip_mpi<Ordinal, BigOrdinal, Communicator>;
#    elif defined( PLATFORM_SYCL )
using current = sycl_mpi<Ordinal, BigOrdinal, Communicator>;
#    endif
#else
template <
    class Ordinal = PLATFORM_ORDINAL, class BigOrdinal = PLATFORM_BIG_ORDINAL,
    class Communicator = scfd::communication::trivial_comm<scfd::memory::host>>
using current = local<Ordinal, BigOrdinal, Communicator>;
#endif

}
}

#endif
