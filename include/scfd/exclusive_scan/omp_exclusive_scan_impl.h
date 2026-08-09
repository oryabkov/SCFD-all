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
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

namespace scfd
{

template <class Ord>
template <class T>
void omp_exclusive_scan<Ord>::operator()( Ord size, const T *input, T *output, T init_val ) const
{
    if ( size <= 0 )
    {
        return;
    }

#ifndef _OPENMP
    T sum = init_val;
    for ( Ord i = 0; i < size; ++i )
    {
        const T value = input[i];
        output[i]     = sum;
        sum += value;
    }
#else
    std::vector<T> partial_sums;

#    pragma omp parallel
    {
        const int thread_id     = omp_get_thread_num();
        const int threads_count = omp_get_num_threads();

#    pragma omp single
        {
            partial_sums.assign( threads_count + 1, T( 0 ) );
        }

        T local_sum = T( 0 );
#    pragma omp for schedule( static )
        for ( Ord i = 0; i < size; ++i )
        {
            const T value = input[i];
            output[i]     = local_sum;
            local_sum += value;
        }

        partial_sums[thread_id + 1] = local_sum;

#    pragma omp barrier
#    pragma omp single
        {
            partial_sums[0] = init_val;
            for ( int i = 1; i <= threads_count; ++i )
            {
                partial_sums[i] += partial_sums[i - 1];
            }
        }

        const T offset = partial_sums[thread_id];
#    pragma omp for schedule( static )
        for ( Ord i = 0; i < size; ++i )
        {
            output[i] += offset;
        }
    }
#endif
}

}

#endif
