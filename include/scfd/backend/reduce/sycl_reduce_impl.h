// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_SYCL_REDUCE_IMPL_H__
#define __SCFD_SYCL_REDUCE_IMPL_H__

#include "sycl_reduce.h"
#include "scfd/utils/init_sycl.h"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord>
template <class T>
T sycl_reduce<Ord>::operator()( Ord size, const T *input, T init_val ) const
{
    return operator()( size, input, init_val, functional::plus<T>() );
}

template <class Ord>
template <class T, class BinaryOp>
T sycl_reduce<Ord>::operator()( Ord size, const T *input, T init_val, BinaryOp binary_op ) const
{
    auto policy = dpl::execution::make_device_policy( sycl_device_queue );
    return dpl::reduce( policy, input, input + size, init_val, binary_op );
}

}

#endif
