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

#ifndef __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_DIMS_H__
#define __SCFD_ARRAYS_NESTED_INITIALIZER_LIST_DIMS_H__

#include <initializer_list>
#include "nested_initializer_list_gen.h"

/// TODO add template Ord intead of int as in other classes??

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class T, int DimsNum, int CurrIdx, bool End>
struct nested_initializer_list_dims_
{
};

template<class T, int DimsNum, int CurrIdx>
struct nested_initializer_list_dims_<T,DimsNum,CurrIdx,true>
{
    template<class DimsVec>
    static void calc(std::initializer_list<T> il, DimsVec &dims)
    {
        dims[DimsNum-1] = il.size();
    }
    template<class DimsVec>
    static bool check_if_square(std::initializer_list<T> il, DimsVec &dims)
    {
        return dims[DimsNum-1] == il.size();
    }
};

template<class T, int DimsNum, int CurrIdx>
struct nested_initializer_list_dims_<T,DimsNum,CurrIdx,false>
{
    typedef nested_initializer_list_dims_<T,DimsNum,CurrIdx+1,CurrIdx+1==DimsNum-1> next_dims_t;

    template<class DimsVec>
    static void calc(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum-CurrIdx>::type> il,
        DimsVec &dims
    )
    {
        if (il.size() == 0)
        {
            throw std::logic_error("nested_initializer_list_dims_::empty initializer list case is not implemented - TODO!!");
        }
        dims[CurrIdx] = il.size();
        next_dims_t::calc(*il.begin(),dims);
    }
    template<class DimsVec>
    static bool check_if_square(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum-CurrIdx>::type> il,
        DimsVec &dims
    )
    {
        if (dims[CurrIdx] != il.size()) return false;
        for (ordinal_type i1 = 0;i1 < il.size();++i1)
        {
            if (!next_dims_t::check_if_square(*(il.begin()+i1),dims)) return false;
        }
        return true;
    }
};


template<class T, int DimsNum>
struct nested_initializer_list_dims
{
    typedef nested_initializer_list_dims_<T,DimsNum,0,0==DimsNum-1> dims_t;

    template<class DimsVec>
    static void calc(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum>::type> il,
        DimsVec &dims
    )
    {
        dims_t::calc(il,dims);
    }
    template<class DimsVec>
    static bool check_if_square(
        std::initializer_list<typename nested_initializer_list_gen<T,DimsNum>::type> il,
        DimsVec &dims
    )
    {
        return dims_t::check_if_square(il,dims);
    }
};

}
}
}

#endif
