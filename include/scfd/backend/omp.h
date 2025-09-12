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

#ifndef __SCFD_BACKEND_OMP_H__
#define __SCFD_BACKEND_OMP_H__

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_impl.h>
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/reduce/omp_reduce_impl.h>


namespace scfd
{
namespace backend
{

struct omp
{
    using memory_type       = scfd::memory::host;
    template <class Ordinal = int>
    using for_each_type     = scfd::for_each::omp<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type  = scfd::for_each::omp_nd<Dim, Ordinal>;
    using reduce_type       = scfd::omp_reduce<>;   
};

}
}

#endif // __SCFD_BACKEND_OMP_H__