// Copyright © 2016-2021 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_REDUCE_THRUST_H__
#define __SCFD_REDUCE_THRUST_H__

#include "reduce_config.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct thrust_reduce
{
    template <class T>
    T operator()( Ord size, const T *input, T init_val ) const
    {
        return operator()( size, input, init_val, functional::plus<T>() );
    }

    template <class T, class BinaryOp>
    T operator()( Ord size, const T *input, T init_val, BinaryOp binary_op ) const
    {
        ::thrust::device_ptr<const T> input_begin = ::thrust::device_pointer_cast( input ),
                                      input_end   = input_begin + size;
        return ::thrust::reduce( input_begin, input_end, init_val, binary_op );
    }
    void wait() const
    {
    }
};

}

#endif
