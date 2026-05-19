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

#ifndef __SCFD_OMP_SORT_BY_KEY_IMPL_H__
#define __SCFD_OMP_SORT_BY_KEY_IMPL_H__

#include "omp_sort_by_key.h"
#include <scfd/backend/sort_by_key/serial_cpu.h>

namespace scfd
{

template <class Ord>
template <class Key, class Value, class Compare>
void omp_sort_by_key<Ord>::operator()( Ord size, Key *keys, Value *values, Compare compare ) const
{
#pragma omp parallel
    {
#pragma omp single
        {
            detail::sort_by_key_host_impl( size, keys, values, compare );
        }
    }
}

}

#endif
