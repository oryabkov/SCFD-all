// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_OMP_SET_INTERSECTION_IMPL_H__
#define __SCFD_OMP_SET_INTERSECTION_IMPL_H__

#include "omp_set_intersection.h"
#include <scfd/backend/set_intersection/serial_cpu.h>

namespace scfd
{

template <class Ord>
template <class T>
Ord omp_set_intersection<Ord>::operator()( Ord size1, const T *set1, Ord size2, const T *set2, T *result ) const
{
    Ord out_size = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            serial_cpu_set_intersection<Ord> serial_op;
            out_size = serial_op( size1, set1, size2, set2, result );
        }
    }
    return out_size;
}

}

#endif
