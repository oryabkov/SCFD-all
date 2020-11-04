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

#ifndef __SCFD_ARRAYS_SIZE_CALCULATOR_H__
#define __SCFD_ARRAYS_SIZE_CALCULATOR_H__

#include <scfd/static_vec/vec.h>
#include "dim_getter.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

using static_vec::vec;

template<class Ord, Ord Ind, bool End, Ord... Dims>
struct size_calculator_
{
};

template<class Ord, Ord Ind, Ord... Dims>
struct size_calculator_<Ord,Ind,true,Dims...>
{
    template<Ord sz>
    static Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return 1;
    }
};

template<class Ord, Ord Ind, Ord... Dims>
struct size_calculator_<Ord,Ind,false,Dims...>
{
    template<Ord sz>
    static Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return dim_getter<Ord,Ind,Dims...>::get(dyn_sizes)*size_calculator_<Ord,Ind+1,sizeof...(Dims)==Ind+1,Dims...>::get(dyn_sizes);
    }
};

template<class Ord, Ord... Dims>
struct size_calculator
{
    template<Ord sz>
    static Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return size_calculator_<Ord,0,sizeof...(Dims)==0,Dims...>::get(dyn_sizes);
    }
};

}
}
}

#endif
