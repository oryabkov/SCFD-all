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

#ifndef __SCFD_COUNT_BY_KEY_THRUST_H__
#define __SCFD_COUNT_BY_KEY_THRUST_H__

#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct thrust_count_by_key
{
    template <class Key, class Count, class KeyEqual>
    Ord operator()( Ord size, const Key *keys_in, Key *keys_out, Count *counts_out, KeyEqual key_equal ) const
    {
        if ( size <= 0 )
            return 0;

        ::thrust::device_ptr<const Key> keys_begin  = ::thrust::device_pointer_cast( keys_in );
        ::thrust::device_ptr<Key>       keys_out_it = ::thrust::device_pointer_cast( keys_out );
        ::thrust::device_ptr<Count>     counts_it   = ::thrust::device_pointer_cast( counts_out );
        auto                            ones        = ::thrust::make_constant_iterator( Count( 1 ) );
        auto                            new_end =
            ::thrust::reduce_by_key( keys_begin, keys_begin + size, ones, keys_out_it, counts_it, key_equal );
        return static_cast<Ord>( new_end.first - keys_out_it );
    }

    template <class Key, class Count>
    Ord operator()( Ord size, const Key *keys_in, Key *keys_out, Count *counts_out ) const
    {
        return operator()( size, keys_in, keys_out, counts_out, functional::equal_to<Key>() );
    }

    void wait() const
    {
    }
};

}

#endif
