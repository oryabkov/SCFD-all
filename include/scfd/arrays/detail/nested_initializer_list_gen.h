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

#ifndef __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_GEN_H__
#define __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_GEN_H__

#include <initializer_list>

/// NOTE that it generates one less level of nesting than Lev
/// TODO add template Ord intead of int as in other classes??

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class T, int Lev, bool End>
struct nested_initializer_list_gen_
{
};

template<class T, int Lev>
struct nested_initializer_list_gen_<T,Lev,true>
{
    typedef T type;
};

template<class T, int Lev>
struct nested_initializer_list_gen_<T,Lev,false>
{
    typedef std::initializer_list<typename nested_initializer_list_gen_<T,Lev-1,Lev-1==1>::type> type;
};


template<class T, int Lev>
struct nested_initializer_list_gen
{
    typedef typename nested_initializer_list_gen_<T,Lev,Lev==1>::type type;
};

}
}
}

#endif
