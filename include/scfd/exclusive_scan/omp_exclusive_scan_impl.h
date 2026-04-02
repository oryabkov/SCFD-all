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

#ifndef __SCFD_OMP_EXCLUSIVE_SCAN_IMPL_H__
#define __SCFD_OMP_EXCLUSIVE_SCAN_IMPL_H__

#include "omp_exclusive_scan.h"

namespace scfd
{

template <class Ord>
template <class T>
void omp_exclusive_scan<Ord>::operator()( Ord size, const T *input, T *output, T init_val ) const
{
    #pragma omp parallel
    {
        T sum = init_val;
        #pragma omp for nowait
        for ( Ord i = 0; i < size; ++i )
        {
            output[i] = sum;
            sum += input[i];
        }
    }
}

}

#endif
