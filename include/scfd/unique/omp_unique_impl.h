// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_OMP_UNIQUE_IMPL_H__
#define __SCFD_OMP_UNIQUE_IMPL_H__

#include "omp_unique.h"
#include <algorithm>

namespace scfd
{

template <class Ord>
template <class T>
Ord omp_unique<Ord>::operator()( Ord size, T *data ) const
{
    Ord result = size;
    #pragma omp parallel
    {
        #pragma omp single
        {
            auto end = std::unique( data, data + size );
            result = static_cast<Ord>( end - data );
        }
    }
    return result;
}

}

#endif
