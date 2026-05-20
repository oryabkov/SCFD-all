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

#ifndef __SCFD_OMP_COUNT_BY_KEY_IMPL_H__
#define __SCFD_OMP_COUNT_BY_KEY_IMPL_H__

#include "omp_count_by_key.h"
#include <omp.h>
#include <cstddef>
#include <vector>

namespace scfd
{
namespace detail
{

template <class Ord>
Ord omp_count_by_key_chunk_begin( Ord size, int tid, int threads )
{
    return static_cast<Ord>(
        ( static_cast<std::size_t>( size ) * static_cast<std::size_t>( tid ) ) / static_cast<std::size_t>( threads )
    );
}

}

template <class Ord>
template <class Key, class Count, class KeyEqual>
Ord omp_count_by_key<Ord>::operator()(
    Ord size, const Key *keys_in, Key *keys_out, Count *counts_out, KeyEqual key_equal
) const
{
    if ( size <= 0 )
        return 0;

    const int        max_threads = omp_get_max_threads();
    int              threads     = 1;
    std::vector<Ord> counts( static_cast<std::size_t>( max_threads ), 0 );
    std::vector<Ord> offsets( static_cast<std::size_t>( max_threads ) + 1, 0 );

#pragma omp parallel
    {
        const int tid       = omp_get_thread_num();
        const int nthreads  = omp_get_num_threads();
        const Ord chunk_beg = detail::omp_count_by_key_chunk_begin( size, tid, nthreads );
        const Ord chunk_end = detail::omp_count_by_key_chunk_begin( size, tid + 1, nthreads );

#pragma omp single
        {
            threads = nthreads;
        }

        Ord count = 0;
        for ( Ord i = chunk_beg; i < chunk_end; ++i )
        {
            if ( i == 0 || !key_equal( keys_in[i - 1], keys_in[i] ) )
                ++count;
        }
        counts[static_cast<std::size_t>( tid )] = count;
    }

    for ( int i = 0; i < threads; ++i )
        offsets[static_cast<std::size_t>( i + 1 )] =
            offsets[static_cast<std::size_t>( i )] + counts[static_cast<std::size_t>( i )];

#pragma omp parallel num_threads( threads )
    {
        const int tid       = omp_get_thread_num();
        const Ord chunk_beg = detail::omp_count_by_key_chunk_begin( size, tid, threads );
        const Ord chunk_end = detail::omp_count_by_key_chunk_begin( size, tid + 1, threads );
        Ord       out       = offsets[static_cast<std::size_t>( tid )];

        for ( Ord i = chunk_beg; i < chunk_end; ++i )
        {
            if ( i != 0 && key_equal( keys_in[i - 1], keys_in[i] ) )
                continue;

            const Key key = keys_in[i];
            Ord       j   = i + 1;
            while ( j < size && key_equal( key, keys_in[j] ) )
                ++j;

            keys_out[out]   = key;
            counts_out[out] = static_cast<Count>( j - i );
            ++out;
        }
    }

    return offsets[static_cast<std::size_t>( threads )];
}

}

#endif
