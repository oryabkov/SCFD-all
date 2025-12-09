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

#ifndef __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_FILL_H__
#define __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_FILL_H__

#include <initializer_list>
#include "index_sequence.h"
#include "nested_initializer_list_gen.h"

/// TODO add template Ord intead of int as in other classes??

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class T, int DimsNum, int CurrIdx, bool End>
struct nested_initializer_list_fill_
{
};

template<class T, int DimsNum, int CurrIdx>
struct nested_initializer_list_fill_<T,DimsNum,CurrIdx,true>
{
    template<class OtherArray,class IndexVec,ordinal_type... I>
    static T   &index_get_(const OtherArray &a, const IndexVec &idx, 
                           detail::index_sequence<ordinal_type,I...>)
    {
        return a.operator()(idx[I]...);
    }
    template<class OtherArray,class IndexVec>
    static void fill(std::initializer_list<T> il, const OtherArray &host_buf, IndexVec &idx)
    {
        for (ordinal_type i1 = 0;i1 < il.size();++i1)
        {
            idx[DimsNum-1] = i1;
            index_get_(host_buf,idx,detail::make_index_sequence<ordinal_type,DimsNum>{}) = 
                *(il.begin()+i1);
        }
    }
};

template<class T, int DimsNum, int CurrIdx>
struct nested_initializer_list_fill_<T,DimsNum,CurrIdx,false>
{
    typedef nested_initializer_list_fill_<T,DimsNum,CurrIdx+1,CurrIdx+1==DimsNum-1> next_fill_t;

    template<class OtherArray,class IndexVec>
    static void fill(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum-CurrIdx>::type> il,
        const OtherArray &host_buf, IndexVec &idx
    )
    {
        for (ordinal_type i1 = 0;i1 < il.size();++i1)
        {
            idx[CurrIdx] = i1;
            next_fill_t::fill(*(il.begin()+i1),host_buf,idx);
        }
    }
};


template<class T, int DimsNum>
struct nested_initializer_list_fill
{
    typedef nested_initializer_list_fill_<T,DimsNum,0,0==DimsNum-1> fill_t;

    template<class OtherArray,class IndexVec>
    static void fill(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum>::type> il,
        const OtherArray &host_buf, IndexVec &idx
    )
    {
        fill_t::fill(il,host_buf,idx);
    }
};

}
}
}

#endif
