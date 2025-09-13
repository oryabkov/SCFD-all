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

#if  defined(PLATFORM_SERIAL_CPU)
#include "serial_cpu.h"
namespace scfd
{
namespace backend
{
using selection = serial_cpu;
}
}

#elif defined(PLATFORM_OMP)
#include "omp.h"
namespace scfd
{
namespace backend
{
using selection = omp;
}
}

#elif defined(PLATFORM_CUDA)
#include "cuda.h"
namespace scfd
{
namespace backend
{
using selection = cuda;
}
}

#elif defined(PLATFORM_HIP)
#include "hip.h"
namespace scfd
{
namespace backend
{
using selection = hip;
}
}

#elif defined(PLATFORM_SYCL)
#include "sycl.h"
namespace scfd
{
namespace backend
{
using selection = sycl;
}
}

#else
#error "No platform has been chosen for backend"

#endif

namespace scfd
{
namespace backend
{
// usefull aliases
using memory      = selection::memory_type;
template <class Ordinal = int>
using for_each    = selection::for_each_type<Ordinal>;
template <int Dim, class Ordinal = int>
using for_each_nd = selection::for_each_nd_type<Dim, Ordinal>;
using reduce      = selection::reduce_type;
}
}

#endif
