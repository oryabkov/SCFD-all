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

#ifndef __SCFD_SET_INTERSECTION_THRUST_H__
#define __SCFD_SET_INTERSECTION_THRUST_H__

#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>

namespace scfd
{

template <class Ord = int>
struct thrust_set_intersection
{
    template <class T>
    Ord operator()( Ord size1, const T *set1, Ord size2, const T *set2, T *result ) const
    {
        if ( size1 <= 0 || size2 <= 0 )
            return 0;
        ::thrust::device_ptr<const T> set1_begin   = ::thrust::device_pointer_cast( set1 );
        ::thrust::device_ptr<const T> set2_begin   = ::thrust::device_pointer_cast( set2 );
        ::thrust::device_ptr<T>       result_begin = ::thrust::device_pointer_cast( result );
        auto                          end =
            ::thrust::set_intersection( set1_begin, set1_begin + size1, set2_begin, set2_begin + size2, result_begin );
        return static_cast<Ord>( end - result_begin );
    }
    void wait() const
    {
    }
};

}

#endif
