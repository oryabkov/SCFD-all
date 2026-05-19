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

#ifndef __SCFD_SERIAL_CPU_REDUCE_BY_KEY_H__
#define __SCFD_SERIAL_CPU_REDUCE_BY_KEY_H__

#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{
namespace detail
{

template <class Ord, class Key, class Value, class KeyEqual, class BinaryOp>
Ord reduce_by_key_host_impl(
    Ord size, const Key *keys_in, const Value *values_in, Key *keys_out, Value *values_out, KeyEqual key_equal,
    BinaryOp binary_op
)
{
    if ( size <= 0 )
        return 0;

    Ord   out_size    = 0;
    Key   current_key = keys_in[0];
    Value current_val = values_in[0];

    for ( Ord i = 1; i < size; ++i )
    {
        if ( key_equal( current_key, keys_in[i] ) )
        {
            current_val = binary_op( current_val, values_in[i] );
        }
        else
        {
            keys_out[out_size]   = current_key;
            values_out[out_size] = current_val;
            ++out_size;
            current_key = keys_in[i];
            current_val = values_in[i];
        }
    }

    keys_out[out_size]   = current_key;
    values_out[out_size] = current_val;
    ++out_size;

    return out_size;
}

}

template <class Ord = int>
struct serial_cpu_reduce_by_key
{
    template <class Key, class Value, class KeyEqual, class BinaryOp>
    Ord operator()(
        Ord size, const Key *keys_in, const Value *values_in, Key *keys_out, Value *values_out, KeyEqual key_equal,
        BinaryOp binary_op
    ) const
    {
        return detail::reduce_by_key_host_impl(
            size, keys_in, values_in, keys_out, values_out, key_equal, binary_op
        );
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
