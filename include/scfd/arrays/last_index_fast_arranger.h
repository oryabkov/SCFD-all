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

#ifndef __SCFD_ARRAYS_LAST_INDEX_FAST_ARRANGER_H__
#define __SCFD_ARRAYS_LAST_INDEX_FAST_ARRANGER_H__

#include "arrays_config.h"
#include <scfd/static_vec/vec.h>
#include "arranger_base.h"

namespace scfd
{
namespace arrays
{

using static_vec::vec;

template<ordinal_type... Dims>
struct last_index_fast_arranger : public arranger_base<ordinal_type,Dims...>
{
};

template<ordinal_type Dim0>
struct last_index_fast_arranger<Dim0> : public arranger_base<ordinal_type,Dim0>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return i0 - this->template get_index0<0>();
#else
        return i0;   
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1>
struct last_index_fast_arranger<Dim0,Dim1> : public arranger_base<ordinal_type,Dim0,Dim1>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                               (i0 - this->template get_index0<0>())*
                this->template get_dim<1>() + i1 - this->template get_index0<1>();
#else
        return                               (i0)*
                this->template get_dim<1>() + i1;
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2>
struct last_index_fast_arranger<Dim0,Dim1,Dim2> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                              ((i0 - this->template get_index0<0>())*
                this->template get_dim<1>() + i1 - this->template get_index0<1>())*
                this->template get_dim<2>() + i2 - this->template get_index0<2>();
#else
        return                              ((i0)*
                this->template get_dim<1>() + i1)*
                this->template get_dim<2>() + i2;
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3>
struct last_index_fast_arranger<Dim0,Dim1,Dim2,Dim3> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2, 
                                               ordinal_type i3)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                            (((i0 - this->template get_index0<0>())*
               this->template get_dim<1>() + i1 - this->template get_index0<1>())*
               this->template get_dim<2>() + i2 - this->template get_index0<2>())*
               this->template get_dim<3>() + i3 - this->template get_index0<3>();
#else
        return                            (((i0)*
               this->template get_dim<1>() + i1)*
               this->template get_dim<2>() + i2)*
               this->template get_dim<3>() + i3;
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3, ordinal_type Dim4>
struct last_index_fast_arranger<Dim0,Dim1,Dim2,Dim3,Dim4> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3,Dim4>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2, 
                                               ordinal_type i3, 
                                               ordinal_type i4)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                           ((((i0 - this->template get_index0<0>())*
               this->template get_dim<1>() + i1 - this->template get_index0<1>())*
               this->template get_dim<2>() + i2 - this->template get_index0<2>())*
               this->template get_dim<3>() + i3 - this->template get_index0<3>())*
               this->template get_dim<4>() + i4 - this->template get_index0<4>();
#else
        return                           ((((i0)*
               this->template get_dim<1>() + i1)*
               this->template get_dim<2>() + i2)*
               this->template get_dim<3>() + i3)*
               this->template get_dim<4>() + i4;
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3, ordinal_type Dim4, ordinal_type Dim5>
struct last_index_fast_arranger<Dim0,Dim1,Dim2,Dim3,Dim4,Dim5> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3,Dim4,Dim5>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2, 
                                               ordinal_type i3, 
                                               ordinal_type i4, 
                                               ordinal_type i5)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                          (((((i0 - this->template get_index0<0>())*
               this->template get_dim<1>() + i1 - this->template get_index0<1>())*
               this->template get_dim<2>() + i2 - this->template get_index0<2>())*
               this->template get_dim<3>() + i3 - this->template get_index0<3>())*
               this->template get_dim<4>() + i4 - this->template get_index0<4>())*
               this->template get_dim<5>() + i5 - this->template get_index0<5>();
#else
        return                          (((((i0)*
               this->template get_dim<1>() + i1)*
               this->template get_dim<2>() + i2)*
               this->template get_dim<3>() + i3)*
               this->template get_dim<4>() + i4)*
               this->template get_dim<5>() + i5;
#endif
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3, ordinal_type Dim4, ordinal_type Dim5, ordinal_type Dim6>
struct last_index_fast_arranger<Dim0,Dim1,Dim2,Dim3,Dim4,Dim5,Dim6> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3,Dim4,Dim5,Dim6>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2, 
                                               ordinal_type i3, 
                                               ordinal_type i4, 
                                               ordinal_type i5, 
                                               ordinal_type i6)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        return                         ((((((i0 - this->template get_index0<0>())*
               this->template get_dim<1>() + i1 - this->template get_index0<1>())*
               this->template get_dim<2>() + i2 - this->template get_index0<2>())*
               this->template get_dim<3>() + i3 - this->template get_index0<3>())*
               this->template get_dim<4>() + i4 - this->template get_index0<4>())*
               this->template get_dim<5>() + i5 - this->template get_index0<5>())*
               this->template get_dim<6>() + i6 - this->template get_index0<6>();
#else
        return                         ((((((i0)*
               this->template get_dim<1>() + i1)*
               this->template get_dim<2>() + i2)*
               this->template get_dim<3>() + i3)*
               this->template get_dim<4>() + i4)*
               this->template get_dim<5>() + i5)*
               this->template get_dim<6>() + i6;
#endif
    }
};

/*template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type Dim3, ordinal_type Dim4, ordinal_type Dim5, ordinal_type Dim6>
struct last_index_fast_arranger<Dim0,Dim1,Dim2,Dim3,Dim4,Dim5,Dim6> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2,Dim3,Dim4,Dim5,Dim6>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, ordinal_type i1, ordinal_type i2, ordinal_type i3, ordinal_type i4, ordinal_type i5, ordinal_type i6)
    {
        return (((((i6*this->template get_dim<5>() + i5)*
                       this->template get_dim<4>() + i4)*
                       this->template get_dim<3>() + i3)*
                       this->template get_dim<2>() + i2)*
                       this->template get_dim<1>() + i1)*
                       this->template get_dim<0>() + i0;
    }
};*/

}
}

#endif
