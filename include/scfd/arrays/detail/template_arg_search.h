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

#ifndef __SCFD_ARRAYS_TEMPLATE_ARG_SEARCH_H__
#define __SCFD_ARRAYS_TEMPLATE_ARG_SEARCH_H__

#include <type_traits>

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class Ord, bool Found, class C, class... Args>
struct template_arg_search_
{
};

template<class Ord, class C, class... Args>
struct template_arg_search_<Ord,true,C,Args...>
{
    static const Ord value = 0;
};

template<class Ord, class C, class Head1, class Head2, class... Tail>
struct template_arg_search_<Ord,false,C,Head1,Head2,Tail...>
{
    static const Ord value = 1 + template_arg_search_<Ord,std::is_same<C,Head2>::value,C,Head2,Tail...>::value;
};

template<class Ord, class C, class Head1>
struct template_arg_search_<Ord,false,C,Head1>
{
    static_assert(sizeof(Ord) != sizeof(Ord),"template_arg_search_::seems needed argument is not here");
};

template<class Ord, class C, class Head, class... Tail>
struct template_arg_search
{
    static const Ord value = template_arg_search_<Ord,std::is_same<C,Head>::value,C,Head,Tail...>::value;
};

}
}
}

#endif
