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
#include <scfd/reduce/sycl_reduce_impl.h>

#define MAKE_SYCL_DEVICE_COPYABLE(kernel) template<>    \
struct sycl::is_device_copyable<typename kernel>        \
    : std::true_type {}

namespace scfd
{
namespace backend
{
struct sycl
{
    using memory_type       = scfd::memory::sycl_device;
    template <class Ordinal = int>
    using for_each_type     = scfd::for_each::sycl<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type  = scfd::for_each::sycl_nd<Dim, Ordinal>;
    using reduce_type       = scfd::sycl_reduce<>;
};
}
}

#endif // __SCFD_BACKEND_SYCL_H__

