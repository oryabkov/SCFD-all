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

#ifndef __SCFD_SERIAL_CPU_SET_INTERSECTION_H__
#define __SCFD_SERIAL_CPU_SET_INTERSECTION_H__

#include <algorithm>

namespace scfd
{

template <class Ord = int>
struct serial_cpu_set_intersection
{
    template <class T>
    Ord operator()( Ord size1, const T *set1, Ord size2, const T *set2, T *result ) const
    {
        if ( size1 <= 0 || size2 <= 0 )
            return 0;
        auto end = std::set_intersection( set1, set1 + size1, set2, set2 + size2, result );
        return static_cast<Ord>( end - result );
    }
    void wait() const
    {
    }
};

}

#endif
