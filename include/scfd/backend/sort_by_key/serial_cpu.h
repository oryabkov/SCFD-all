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

#ifndef __SCFD_SERIAL_CPU_SORT_BY_KEY_H__
#define __SCFD_SERIAL_CPU_SORT_BY_KEY_H__

#include <algorithm>
#include <vector>
#include <scfd/backend/functional/basic_ops.h>
#include <scfd/backend/value_pair.h>

namespace scfd
{
namespace detail
{

template <class Ord, class Key, class Value, class Compare>
void sort_by_key_host_impl( Ord size, Key *keys, Value *values, Compare compare )
{
    if ( size <= 0 )
        return;

    using pair_type = scfd::backend::value_pair<Key, Value>;
    std::vector<pair_type> pairs;
    pairs.reserve( static_cast<size_t>( size ) );
    for ( Ord i = 0; i < size; ++i )
    {
        pairs.push_back( pair_type( keys[i], values[i] ) );
    }
    std::sort( pairs.begin(), pairs.end(), [compare]( const pair_type &a, const pair_type &b ) {
        return compare( a.first, b.first );
    } );
    for ( Ord i = 0; i < size; ++i )
    {
        keys[i]   = pairs[static_cast<size_t>( i )].first;
        values[i] = pairs[static_cast<size_t>( i )].second;
    }
}

}

template <class Ord = int>
struct serial_cpu_sort_by_key
{
    template <class Key, class Value, class Compare>
    void operator()( Ord size, Key *keys, Value *values, Compare compare ) const
    {
        if ( size <= 0 )
            return;
        detail::sort_by_key_host_impl( size, keys, values, compare );
    }

    template <class Key, class Value>
    void operator()( Ord size, Key *keys, Value *values ) const
    {
        operator()( size, keys, values, functional::less<Key>() );
    }

    void wait() const
    {
    }
};

}

#endif
