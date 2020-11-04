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

#ifndef __SCFD_ARRAYS_DYN_DIM_COUNTER_H__
#define __SCFD_ARRAYS_DYN_DIM_COUNTER_H__

#include "../arrays_config.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class Ord, Ord Ind, bool End, Ord... Dims>
struct dyn_dim_counter_
{
};

template<class Ord, Ord Ind, Ord... Dims>
struct dyn_dim_counter_<Ord,Ind,true,Dims...>
{
    static const Ord value = 0;
};

template<class Ord, Ord Ind, Ord Dim1, Ord... Dims>
struct dyn_dim_counter_<Ord,Ind,false,Dim1,Dims...>
{
    static const Ord value = dyn_dim_counter_<Ord,Ind-1,Ind==1,Dims...>::value + (Dim1 == dyn_dim);
};

template<class Ord, Ord Ind, Ord... Dims>
struct dyn_dim_counter
{
    static const Ord value = dyn_dim_counter_<Ord,Ind,Ind==0,Dims...>::value;
};

}
}
}

#endif
