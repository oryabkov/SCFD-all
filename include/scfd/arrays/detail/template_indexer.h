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

#ifndef __SCFD_ARRAYS_TEMPLATE_INDEXER_H__
#define __SCFD_ARRAYS_TEMPLATE_INDEXER_H__

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class Ord, Ord Ind, bool End, Ord... Dims>
struct template_indexer_
{
};

template<class Ord, Ord Ind, Ord Dim1, Ord... Dims>
struct template_indexer_<Ord,Ind,true,Dim1,Dims...>
{
    static const Ord value = Dim1;
};

template<class Ord, Ord Ind, Ord Dim1, Ord... Dims>
struct template_indexer_<Ord,Ind,false,Dim1,Dims...>
{
    static const Ord value = template_indexer_<Ord,Ind-1,Ind==1,Dims...>::value;
};

template<class Ord, Ord Ind, Ord... Dims>
struct template_indexer
{
    static const Ord value = template_indexer_<Ord,Ind,Ind==0,Dims...>::value;
};

}
}
}

#endif
