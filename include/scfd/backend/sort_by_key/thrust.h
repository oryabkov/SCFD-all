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

#ifndef __SCFD_SORT_BY_KEY_THRUST_H__
#define __SCFD_SORT_BY_KEY_THRUST_H__

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct thrust_sort_by_key
{
    template <class Key, class Value, class Compare>
    void operator()( Ord size, Key *keys, Value *values, Compare compare ) const
    {
        if ( size <= 0 )
            return;
        ::thrust::device_ptr<Key>   keys_begin   = ::thrust::device_pointer_cast( keys );
        ::thrust::device_ptr<Value> values_begin = ::thrust::device_pointer_cast( values );
        ::thrust::sort_by_key( keys_begin, keys_begin + size, values_begin, compare );
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
