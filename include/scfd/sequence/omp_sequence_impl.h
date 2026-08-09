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

#ifndef __SCFD_OMP_SEQUENCE_IMPL_H__
#define __SCFD_OMP_SEQUENCE_IMPL_H__

#include "omp_sequence.h"

namespace scfd
{

template <class Ord>
template <class T>
void omp_sequence<Ord>::operator()( Ord size, T *output, T init_val, T step ) const
{
#pragma omp parallel for
    for ( Ord i = 0; i < size; ++i )
    {
        output[i] = init_val + static_cast<T>( i ) * step;
    }
}

}

#endif
