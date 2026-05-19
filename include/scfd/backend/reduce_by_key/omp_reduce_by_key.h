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

#ifndef __SCFD_OMP_REDUCE_BY_KEY_H__
#define __SCFD_OMP_REDUCE_BY_KEY_H__

#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct omp_reduce_by_key
{
    template <class Key, class Value, class KeyEqual, class BinaryOp>
    Ord operator()(
        Ord size, const Key *keys_in, const Value *values_in, Key *keys_out, Value *values_out, KeyEqual key_equal,
        BinaryOp binary_op
    ) const;

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
