// Copyright © 2016-2026 Evstigneev Nikolay Mikhaylovitch, Ryabkov Oleg Igorevich

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
// 
// 
#ifndef __SCFD_ARRAYS_CUSTOM_INDEX_FAST_ARRANGER_H__
#define __SCFD_ARRAYS_CUSTOM_INDEX_FAST_ARRANGER_H__

#include <scfd/static_vec/vec.h>
#include <scfd/arrays/arranger_base.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/todo.h>

namespace scfd
{
namespace arrays
{

using static_vec::vec;

namespace detail
{

template<ordinal_type I>
struct axis_size 
{
    template<typename... Ts>
    __DEVICE_TAG__ static ordinal_type get(const Ts&... sizes) 
    {
        SCFD_ATODO("TODO! Implement on demand!");
    }
};

template<>
struct axis_size<0>
{
    template<typename... Ts>
    __DEVICE_TAG__ static ordinal_type get(ordinal_type i0, const Ts&... index_tail) 
    {
        return i0; 
    }
};

template<>
struct axis_size<1>
{
    template<typename... Ts>
    __DEVICE_TAG__ static ordinal_type get(ordinal_type i0, ordinal_type i1, const Ts&... index_tail) 
    {
        return i1; 
    }
};

template<>
struct axis_size<2>
{
    template<typename... Ts>
    __DEVICE_TAG__ static ordinal_type get(ordinal_type i0, ordinal_type i1, ordinal_type i2, const Ts&... index_tail) 
    {
        return i2; 
    }
};

template<>
struct axis_size<3>
{
    template<typename... Ts>
    __DEVICE_TAG__ static ordinal_type get(ordinal_type i0, ordinal_type i1, ordinal_type i2, ordinal_type i3, const Ts&... index_tail) 
    {
        return i3; 
    }
};

}


template<ordinal_type... Dims>
struct custom_index_fast_arranger /*: public arranger_base<ordinal_type,Dims...>*/
{
};

template<ordinal_type First, ordinal_type Second>
struct custom_index_fast_arranger<First, Second>
{
    template<ordinal_type Dim0, ordinal_type Dim1>
    struct type : public arranger_base<ordinal_type,Dim0,Dim1>
    {
        __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                                   ordinal_type i1)const
        {

            
            #ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT   
                SCFD_ATODO("TODO! Implement TEST on demand!");             
                const auto first_axis_size = this->template get_dim<First>();
                return detail::axis_size<First>::get(i0, i1) - this->template get_index0<First>() + 
                       first_axis_size*(detail::axis_size<Second>::get(i0, i1) - this->template get_index0<Second>());
            #else
                const auto first_axis_size = this->template get_dim<First>();
                return detail::axis_size<First>::get(i0, i1) + first_axis_size*detail::axis_size<Second>::get(i0, i1);
            #endif
            
        }
    };
};

template<ordinal_type First, ordinal_type Second, ordinal_type Third>
struct custom_index_fast_arranger<First, Second, Third>
{
    template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2>
    struct type : public arranger_base<ordinal_type,Dim0,Dim1,Dim2>
    {
        __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                                   ordinal_type i1, 
                                                   ordinal_type i2)const
        {
            #ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT

            SCFD_ATODO("TODO! Implement TEST on demand!");
            return detail::axis_size<First>::get(i0, i1, i2) - this->template get_index0<First>()  + 
                   this->template get_dim<First>()*( (detail::axis_size<Second>::get(i0, i1, i2) - this->template get_index0<Second>()) +  
                   this->template get_dim<Second>() * (detail::axis_size<Third>::get(i0, i1, i2) - this->template get_index0<Third>()) );

            #else
     
            return detail::axis_size<First>::get(i0, i1, i2)  + 
                   this->template get_dim<First>()*( detail::axis_size<Second>::get(i0, i1, i2) +  
                   this->template get_dim<Second>() * detail::axis_size<Third>::get(i0, i1, i2) );

            #endif
        }
    };
};


template<ordinal_type First, ordinal_type Second, ordinal_type Third, ordinal_type Forth>
struct custom_index_fast_arranger<First, Second, Third, Forth>
{
    template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3>
    struct type : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3>
    {
        __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                                   ordinal_type i1, 
                                                   ordinal_type i2,
                                                   ordinal_type i3)const
        {
            #ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT

            SCFD_ATODO("TODO! Implement TEST on demand!");
            return detail::axis_size<First>::get(i0, i1, i2, i3) - this->template get_index0<First>() + 
                   this->template get_dim<First>()*( (detail::axis_size<Second>::get(i0, i1, i2, i3) - this->template get_index0<Second>()) +  
                   this->template get_dim<Second>()*( (detail::axis_size<Third>::get(i0, i1, i2, i3) - this->template get_index0<Third>()) +  
                   this->template get_dim<Third>()*(detail::axis_size<Forth>::get(i0, i1, i2, i3) - this->template get_index0<Forth>()  )) );

            #else
     
            return detail::axis_size<First>::get(i0, i1, i2, i3) + 
                   this->template get_dim<First>()*( detail::axis_size<Second>::get(i0, i1, i2, i3) +  
                   this->template get_dim<Second>()*( detail::axis_size<Third>::get(i0, i1, i2, i3) +  
                   this->template get_dim<Third>()*detail::axis_size<Forth>::get(i0, i1, i2, i3)  ) );

            #endif
        }
    };
};

}
}


#endif