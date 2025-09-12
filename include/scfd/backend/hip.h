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

#ifndef __SCFD_BACKEND_HIP_H__
#define __SCFD_BACKEND_HIP_H__

#include <scfd/utils/init_hip.h>
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_impl.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

namespace scfd
{
namespace backend
{
struct hip
{
    using memory_type       = scfd::memory::hip_device;
    template <class Ordinal = int>
    using for_each_type     = scfd::for_each::hip<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type  = scfd::for_each::hip_nd<Dim, Ordinal>;
    using reduce_type       = scfd::thrust_reduce<>;
};
}
}

#endif // __SCFD_BACKEND_HIP_H__