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

#ifndef __SCFD_INCLUSIVE_SCAN_SYCL_IMPL_H__
#define __SCFD_INCLUSIVE_SCAN_SYCL_IMPL_H__

#include "sycl.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <scfd/utils/init_sycl.h>

namespace scfd
{
namespace inclusive_scan
{

template <class Ord>
template <class T>
void sycl<Ord>::operator()( Ord size, const T *input, T *output ) const
{
    if ( size <= 0 )
        return;
    auto policy = dpl::execution::make_device_policy( sycl_device_queue );
    dpl::inclusive_scan( policy, input, input + size, output );
}

}
}

#endif
