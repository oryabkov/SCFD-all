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
#        include "serial_cpu_mpi.h"
#    elif defined( PLATFORM_OMP )
#        include "omp_mpi.h"
#    elif defined( PLATFORM_CUDA )
#        include "cuda_mpi.h"
#    elif defined( PLATFORM_HIP )
#        include "hip_mpi.h"
#    elif defined( PLATFORM_SYCL )
#        include "sycl_mpi.h"
#    endif
#else
#    include "trivial.h"
#endif

namespace scfd
{
namespace platform
{

// PLATFORM_MPI is a presence flag. Without it, use a single-rank host
// communicator; backend selection assumes either trivial or scfd mpi communicator.
#ifdef PLATFORM_MPI
#    if defined( PLATFORM_SERIAL_CPU )
using current = serial_cpu_mpi<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#    elif defined( PLATFORM_OMP )
using current = omp_mpi<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#    elif defined( PLATFORM_CUDA )
using current = cuda_mpi<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#    elif defined( PLATFORM_HIP )
using current = hip_mpi<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#    elif defined( PLATFORM_SYCL )
using current = sycl_mpi<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#    endif
#else
using current = trivial<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>;
#endif

}
}

#endif
