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

#ifndef __SCFD_ARRAYS_TEMPLATE_DIMS_VEC_GETTER_H__
#define __SCFD_ARRAYS_TEMPLATE_DIMS_VEC_GETTER_H__

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class Ord, Ord Ind, bool End, Ord... Dims>
struct template_dims_vec_getter_
{
};

template<class Ord, Ord Ind, Ord... Dims>
struct template_dims_vec_getter_<Ord,Ind,true,Dims...>
{
    template<class Vec>
    static void get(Vec &sizes)
    {
    }
};

template<class Ord, Ord Ind, Ord DimsHead, Ord... DimsTail>
struct template_dims_vec_getter_<Ord,Ind,false,DimsHead,DimsTail...>
{
    template<class Vec>
    static void get(Vec &sizes)
    {
        sizes[Ind] = DimsHead;
        template_dims_vec_getter_<Ord,Ind+1,sizeof...(DimsTail)==0,DimsTail...>::get(sizes);
    }
};

template<class Ord, Ord... Dims>
struct template_dims_vec_getter
{
    template<class Vec>
    static void get(Vec &sizes)
    {
        template_dims_vec_getter_<Ord,0,sizeof...(Dims)==0,Dims...>::get(sizes);
    }
};

}
}
}

#endif
