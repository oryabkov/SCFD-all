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

#ifndef __SCFD_ARRAYS_DIM_GETTER_H__
#define __SCFD_ARRAYS_DIM_GETTER_H__

#include <scfd/static_vec/vec.h>
#include "dyn_dim_counter.h"
#include "template_indexer.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

using static_vec::vec;

template<class Ord, Ord Ind, bool UseStaticDim, Ord... Dims>
struct dim_getter_
{
};

template<class Ord, Ord Ind, Ord... Dims>
struct dim_getter_<Ord,Ind,true,Dims...>
{
    template<Ord sz>
    static __DEVICE_TAG__ Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return template_indexer<Ord,Ind,Dims...>::value;
    }
};

template<class Ord, Ord Ind, Ord... Dims>
struct dim_getter_<Ord,Ind,false,Dims...>
{
    template<Ord sz>
    static __DEVICE_TAG__ Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return dyn_sizes[dyn_dim_counter<Ord,Ind,Dims...>::value];
    }
};

template<class Ord, Ord Ind, Ord... Dims>
struct dim_getter
{
    template<Ord sz>
    static __DEVICE_TAG__ Ord get(const vec<Ord,sz> &dyn_sizes)
    {
        return dim_getter_<Ord,Ind,template_indexer<Ord,Ind,Dims...>::value != dyn_dim,Dims...>::get(dyn_sizes);
    }
};

}
}
}

#endif
