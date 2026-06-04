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

#ifndef __SCFD_SYCL_SET_INTERSECTION_IMPL_H__
#define __SCFD_SYCL_SET_INTERSECTION_IMPL_H__

#include "sycl_set_intersection.h"
#include <scfd/utils/init_sycl.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

namespace scfd
{

template <class Ord>
template <class T>
Ord sycl_set_intersection<Ord>::operator()( Ord size1, const T *set1, Ord size2, const T *set2, T *result ) const
{
    if ( size1 <= 0 || size2 <= 0 )
        return 0;
    auto policy = dpl::execution::make_device_policy( sycl_device_queue );
    auto end    = dpl::set_intersection( policy, set1, set1 + size1, set2, set2 + size2, result );
    return static_cast<Ord>( end - result );
}

}

#endif
