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

#ifndef __SCFD_ARRAYS_PARAMETER_INDEXER_H__
#define __SCFD_ARRAYS_PARAMETER_INDEXER_H__

template<int Ind, bool End, int... Dims>
struct access_index_
{
};

template<int Ind, int Dim1, int... DimsTail>
struct access_index_<Ind,true,Dim1,DimsTail...>
{
    template<int>using index_t=int;

    static int get(int i1, index_t<DimsTail>... is_tail)
    {
        return i1;
    }
};

template<int Ind, int Dim1, int... DimsTail>
struct access_index_<Ind,false,Dim1,DimsTail...>
{
    template<int>using index_t=int;

    static int get(int i1, index_t<DimsTail>... is_tail)
    {
        return access_index_<Ind-1,Ind==1,DimsTail...>::get(is_tail...);
    }
};

template<int Ind, int... Dims>
struct access_index
{
    template<int>using index_t=int;

    static int get(index_t<Dims>... is,int test)
    {
        return access_index_<Ind,Ind==0,Dims...>::get(is...);
    }
};

#endif