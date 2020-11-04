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

#ifndef __PLACEHOLDER_GET_HELPER_H__
#define __PLACEHOLDER_GET_HELPER_H__

#include "../placeholder.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class Ord, class Index>
struct placeholder_get_helper { };

template<class Ord>
struct placeholder_get_helper<Ord,placeholder>
{
    static Ord get(placeholder index, Ord placeholder_index) { return placeholder_index; }
};

template<class Ord>
struct placeholder_get_helper<Ord,Ord>
{
    static Ord get(Ord index, Ord placeholder_index) { return index; }
};

}
}
}

#endif
