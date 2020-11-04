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

#ifndef __SCFD_ARRAYS_TENSOR_BASE_ND_GEN_H__
#define __SCFD_ARRAYS_TENSOR_BASE_ND_GEN_H__

#include "../tensor_base.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class T, ordinal_type ND, bool End, class Memory, 
         template <ordinal_type... DimsA> class Arranger, 
         ordinal_type... Dims>
struct tensor_base_nd_gen_
{
};

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... DimsA> class Arranger, 
         ordinal_type... Dims>
struct tensor_base_nd_gen_<T,ND,true,Memory,Arranger,Dims...>
{
    typedef tensor_base<T,Memory,Arranger,Dims...> type;
};

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... DimsA> class Arranger, 
         ordinal_type... Dims>
struct tensor_base_nd_gen_<T,ND,false,Memory,Arranger,Dims...>
{
    typedef typename tensor_base_nd_gen_<T,ND-1,ND-1==0,Memory,Arranger,dyn_dim,Dims...>::type type;
};


template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... DimsA> class Arranger, ordinal_type... TensorDims>
struct tensor_base_nd_gen
{
    typedef typename tensor_base_nd_gen_<T,ND,ND==0,Memory,Arranger,TensorDims...>::type type;
};

}
}
}

#endif
