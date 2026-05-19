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

#ifndef __SCFD_OMP_REDUCE_IMPL_H__
#define __SCFD_OMP_REDUCE_IMPL_H__

#include "omp_reduce.h"
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord>
template <class T>
T omp_reduce<Ord>::operator()( Ord size, const T *input, T init_val ) const
{
    return operator()( size, input, init_val, functional::plus<T>() );
}

template <class Ord>
template <class T, class BinaryOp>
T omp_reduce<Ord>::operator()( Ord size, const T *input, T init_val, BinaryOp binary_op ) const
{
    T res = init_val;
#pragma omp parallel
    {
        T    res_private = T();
        bool has_private = false;
#pragma omp for nowait
        for ( Ord i = 0; i < size; ++i )
        {
            if ( has_private )
                res_private = binary_op( res_private, input[i] );
            else
            {
                res_private = input[i];
                has_private = true;
            }
        }
#pragma omp critical
        {
            if ( has_private )
                res = binary_op( res, res_private );
        }
    }
    return res;
}

}

#endif
