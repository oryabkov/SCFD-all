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
#ifndef __SCFD_CUSTOM_INDEX_ARRANGER_H__
#define __SCFD_CUSTOM_INDEX_ARRANGER_H__

#include <scfd/static_vec/vec.h>
#include <scfd/arrays/arranger_base.h>
#include <scfd/utils/device_tag.h>
#include <tuple>

namespace scfd
{
namespace arrays
{
using static_vec::vec;


//Usage of constexpr in kernels:
// https://stackoverflow.com/questions/40742242/does-cuda-c-not-have-tuples-in-device-code
// I've just tried this out and tuple metaprogramming with std:: (std::tuple, std::get, etc ...) will work in device code with C++14 and expt-relaxed-constexpr enabled (CUDA8+) during compilation (e.g. nvcc -std=c++14 xxxx.cu -o yyyyy --expt-relaxed-constexpr) - CUDA 9 required for C++14, but basic std::tuple should work in CUDA 8 if you are limited to that. Thrust/tuple works but has some drawbacks: limited to 10 items and lacking in some of the std::tuple helper functions (e.g. std::tuple_cat). Because tuples and their related functions are compile-time, expt-relaxed-constexpr should enable your std::tuple to "just work". 
template<size_t I>
struct axis_size 
{
    template<typename... Ts>
    __DEVICE_TAG__ static size_t get(const Ts&... sizes) 
    {
        return std::get<I>(std::forward_as_tuple(sizes...) ); 
    }
};


template<ordinal_type... Dims>
struct custom_index_arranger : public arranger_base<ordinal_type,Dims...>
{
};


template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type First, ordinal_type Second>
struct custom_index_arranger<Dim0, Dim1, First, Second> : public arranger_base<ordinal_type,Dim0,Dim1>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1)const
    {

        
        #ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
                
                const auto first_axis = axis_size<First>::get( this->template get_dim<0>(), this->template get_dim<1>() );
                return first_axis*axis_size<First>::get(i0 - this->template get_index0<0>(), i1 - this->template get_index0<1>()) + axis_size<Second>::get(i0, i1); //(i0 - this->template get_index0<0>())*this->template get_dim<1>() + i1 - this->template get_index0<1>();


        #else
                const auto first_axis = axis_size<Second>::get( this->template get_dim<0>(), this->template get_dim<1>() );
                return first_axis*axis_size<First>::get(i0, i1) + axis_size<Second>::get(i0, i1); //(i0)*this->template get_dim<1>() + i1;
        #endif
        
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type First, ordinal_type Second, ordinal_type Third>
struct custom_index_arranger<Dim0,Dim1,Dim2,First, Second, Third> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        printf("TODO! Implement on demand!");
        exit(0);
        return ((i2 - this->template get_index0<2>())*
                this->template get_dim<1>() + i1 - this->template get_index0<1>())*
                this->template get_dim<0>() + i0 - this->template get_index0<0>();
#else
 
        return axis_size<First>::get(i0, i1, i2) + axis_size<First>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>() )*( axis_size<Second>::get(i0, i1, i2) +  axis_size<Second>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>() )* axis_size<Third>::get(i0, i1, i2) );
        // return  ((i2)*this->template get_dim<1>() + i1)*this->template get_dim<0>() + i0;
#endif
    }
};



template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3, ordinal_type First, ordinal_type Second, ordinal_type Third, ordinal_type Forth>
struct custom_index_arranger<Dim0, Dim1, Dim2, Dim3, First, Second, Third, Forth> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2,
                                               ordinal_type i3)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        printf("TODO! Implement on demand!");
        exit(0);
        return ;
#else
 
        return axis_size<First>::get(i0, i1, i2, i3) + axis_size<First>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>(), this->template get_dim<3>() )*( axis_size<Second>::get(i0, i1, i2, i3) +  axis_size<Second>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>(), this->template get_dim<3>()  )*( axis_size<Third>::get(i0, i1, i2, i3) +  axis_size<Third>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>(), this->template get_dim<3>())*axis_size<Forth>::get(i0, i1, i2, i3)  ) );
        // return  ((i2)*this->template get_dim<1>() + i1)*this->template get_dim<0>() + i0;
#endif
    }
};

}
}


#endif