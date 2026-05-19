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

#ifndef __SCFD_REDUCE_BY_KEY_THRUST_H__
#define __SCFD_REDUCE_BY_KEY_THRUST_H__

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct thrust_reduce_by_key
{
    template <class Key, class Value, class KeyEqual, class BinaryOp>
    Ord operator()(
        Ord size, const Key *keys_in, const Value *values_in, Key *keys_out, Value *values_out, KeyEqual key_equal,
        BinaryOp binary_op
    ) const
    {
        if ( size <= 0 )
            return 0;
        ::thrust::device_ptr<const Key>   keys_begin   = ::thrust::device_pointer_cast( keys_in );
        ::thrust::device_ptr<const Value> values_begin = ::thrust::device_pointer_cast( values_in );
        ::thrust::device_ptr<Key>         keys_out_it  = ::thrust::device_pointer_cast( keys_out );
        ::thrust::device_ptr<Value>       values_out_it = ::thrust::device_pointer_cast( values_out );
        auto new_end = ::thrust::reduce_by_key(
            keys_begin, keys_begin + size, values_begin, keys_out_it, values_out_it, key_equal, binary_op
        );
        return static_cast<Ord>( new_end.first - keys_out_it );
    }

    template <class Key, class Value>
    Ord operator()( Ord size, const Key *keys_in, const Value *values_in, Key *keys_out, Value *values_out ) const
    {
        return operator()(
            size, keys_in, values_in, keys_out, values_out, functional::equal_to<Key>(), functional::plus<Value>()
        );
    }

    void wait() const
    {
    }
};

}

#endif
