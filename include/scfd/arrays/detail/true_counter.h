// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_ARRAYS_TRUE_COUNTER_H__
#define __SCFD_ARRAYS_TRUE_COUNTER_H__

namespace scfd
{
namespace arrays
{
namespace detail
{

template< bool ... b> struct true_counter
{
};

template<> struct true_counter<>
{
    static const int value = 0;
};

template<bool head, bool ... tail> struct true_counter<head,tail...>
{
    static const int value = true_counter<tail...>::value + (head ? 1 : 0);
};

}
}
}

#endif
