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
#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace scfd
{
namespace detail
{

template <class Ord>
Ord omp_set_intersection_chunk_begin( Ord size, int tid, int threads )
{
    return static_cast<Ord>(
        ( static_cast<std::size_t>( size ) * static_cast<std::size_t>( tid ) ) / static_cast<std::size_t>( threads )
    );
}

template <class T>
bool omp_set_intersection_equal( const T &a, const T &b )
{
    return !( a < b ) && !( b < a );
}

template <class Ord, class T>
Ord omp_set_intersection_count_range( const T *set1, Ord size1, const T *set2, Ord size2 )
{
    Ord i     = 0;
    Ord j     = 0;
    Ord count = 0;
    while ( i < size1 && j < size2 )
    {
        if ( set1[i] < set2[j] )
            ++i;
        else if ( set2[j] < set1[i] )
            ++j;
        else
        {
            ++count;
            ++i;
            ++j;
        }
    }
    return count;
}

}

template <class Ord>
template <class T>
Ord omp_set_intersection<Ord>::operator()( Ord size1, const T *set1, Ord size2, const T *set2, T *result ) const
{
    if ( size1 <= 0 || size2 <= 0 )
        return 0;

    int threads = 1;
#pragma omp parallel
    {
#pragma omp single
        {
            threads = omp_get_num_threads();
        }
    }

    std::vector<Ord> set1_offsets( static_cast<std::size_t>( threads ) + 1, 0 );
    std::vector<Ord> set2_offsets( static_cast<std::size_t>( threads ) + 1, 0 );
    std::vector<Ord> counts( static_cast<std::size_t>( threads ), 0 );
    std::vector<Ord> offsets( static_cast<std::size_t>( threads ) + 1, 0 );

    set1_offsets[0]                                   = 0;
    set1_offsets[static_cast<std::size_t>( threads )] = size1;
    for ( int tid = 1; tid < threads; ++tid )
    {
        Ord offset = detail::omp_set_intersection_chunk_begin( size1, tid, threads );
        while ( offset < size1 && detail::omp_set_intersection_equal( set1[offset - 1], set1[offset] ) )
            ++offset;
        set1_offsets[static_cast<std::size_t>( tid )] = offset;
    }

    for ( int tid = 0; tid < threads; ++tid )
    {
        const Ord set1_offset = set1_offsets[static_cast<std::size_t>( tid )];
        set2_offsets[static_cast<std::size_t>( tid )] =
            set1_offset < size1 ? static_cast<Ord>( std::lower_bound( set2, set2 + size2, set1[set1_offset] ) - set2 )
                                : size2;
    }
    set2_offsets[static_cast<std::size_t>( threads )] = size2;

#pragma omp parallel for num_threads( threads )
    for ( int tid = 0; tid < threads; ++tid )
    {
        const Ord set1_begin                    = set1_offsets[static_cast<std::size_t>( tid )];
        const Ord set1_end                      = set1_offsets[static_cast<std::size_t>( tid + 1 )];
        const Ord set2_begin                    = set2_offsets[static_cast<std::size_t>( tid )];
        const Ord set2_end                      = set2_offsets[static_cast<std::size_t>( tid + 1 )];
        counts[static_cast<std::size_t>( tid )] = detail::omp_set_intersection_count_range(
            set1 + set1_begin, set1_end - set1_begin, set2 + set2_begin, set2_end - set2_begin
        );
    }

    for ( int i = 0; i < threads; ++i )
        offsets[static_cast<std::size_t>( i + 1 )] =
            offsets[static_cast<std::size_t>( i )] + counts[static_cast<std::size_t>( i )];

#pragma omp parallel for num_threads( threads )
    for ( int tid = 0; tid < threads; ++tid )
    {
        const Ord set1_begin = set1_offsets[static_cast<std::size_t>( tid )];
        const Ord set1_end   = set1_offsets[static_cast<std::size_t>( tid + 1 )];
        const Ord set2_begin = set2_offsets[static_cast<std::size_t>( tid )];
        const Ord set2_end   = set2_offsets[static_cast<std::size_t>( tid + 1 )];
        std::set_intersection(
            set1 + set1_begin, set1 + set1_end, set2 + set2_begin, set2 + set2_end,
            result + offsets[static_cast<std::size_t>( tid )]
        );
    }

    return offsets[static_cast<std::size_t>( threads )];
}

}

#endif
