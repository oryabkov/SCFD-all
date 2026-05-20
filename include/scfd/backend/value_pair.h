// Copyright © 2016-2026 Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_BACKEND_VALUE_PAIR_H__
#define __SCFD_BACKEND_VALUE_PAIR_H__

#include <scfd/utils/device_tag.h>

namespace scfd
{
namespace backend
{

template <class First, class Second>
struct value_pair
{
    First  first;
    Second second;

    __DEVICE_TAG__ value_pair() : first(), second()
    {
    }

    __DEVICE_TAG__ value_pair( First first_, Second second_ ) : first( first_ ), second( second_ )
    {
    }
};

template <class First, class Second>
__DEVICE_TAG__ value_pair<First, Second> make_value_pair( First first, Second second )
{
    return value_pair<First, Second>( first, second );
}

}
}

#endif
