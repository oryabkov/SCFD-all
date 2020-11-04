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

#ifndef __SCFD_ARRAYS_ARRANGER_BASE_H__
#define __SCFD_ARRAYS_ARRANGER_BASE_H__

#include "arrays_config.h"
#include <type_traits>
#include <scfd/static_vec/vec.h>
#include "detail/bool_array.h"
#include "detail/dyn_dim_counter.h"
#include "detail/template_indexer.h"
#include "detail/dim_getter.h"
#include "detail/index0_getter.h"
#include "detail/size_calculator.h"
#include "detail/has_subscript_operator.h"

namespace scfd
{
namespace arrays
{

using static_vec::vec;

template<class Ord, Ord... Dims>
struct arranger_base
{
public:
    static const Ord dynamic_dims_num = detail::dyn_dim_counter<Ord,sizeof...(Dims),Dims...>::value;

protected:
    vec<Ord,dynamic_dims_num>     dyn_dims_;
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
    vec<Ord,dynamic_dims_num>     dyn_indexes0_;
#endif

    template<class... Args,
             class = typename std::enable_if<sizeof...(Args)==dynamic_dims_num>::type,
             class = typename std::enable_if<detail::check_all_are_true< std::is_integral<Args>::value... >::value>::type>
    void set_dyn_dims(Args ...args)
    {
        dyn_dims_ = vec<Ord,dynamic_dims_num>{args...};
    }
    template<class DynDimsVec, 
             class = typename std::enable_if<detail::has_subscript_operator<DynDimsVec,ordinal_type>::value>::type>
    void set_dyn_dims(const DynDimsVec &dyn_dims)
    {
        dyn_dims_ = vec<Ord,dynamic_dims_num>(dyn_dims);
    }
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
    template<class... Args,
             class = typename std::enable_if<sizeof...(Args)==dynamic_dims_num>::type, 
             class = typename std::enable_if<detail::check_all_are_true< std::is_integral<Args>::value... >::value>::type>
    void set_dyn_indexes0(Args ...args)
    {
        dyn_indexes0_ = vec<Ord,dynamic_dims_num>{args...};
    }
    template<class DynIndexes0Vec, 
             class = typename std::enable_if<detail::has_subscript_operator<DynIndexes0Vec,ordinal_type>::value>::type>
    void set_dyn_indexes0(const DynIndexes0Vec &dyn_indexes0)
    {
        dyn_indexes0_ = vec<Ord,dynamic_dims_num>(dyn_indexes0);
    }
    void set_zero_dyn_indexes0()
    {
        for (Ord j = 0;j < dynamic_dims_num;++j) dyn_indexes0_[j] = 0;
    }
#endif
    void copy_dyn_shape(const arranger_base &a)
    {
        dyn_dims_ = a.dyn_dims_;
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        dyn_indexes0_ = a.dyn_indexes0_;
#endif
    }

public:
    template<Ord Ind>
    __DEVICE_TAG__ Ord get_dim()const
    {
        return detail::dim_getter_<Ord,Ind,detail::template_indexer<Ord,Ind,Dims...>::value != dyn_dim,Dims...>::get(dyn_dims_);
    }
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
    template<Ord Ind>
    __DEVICE_TAG__ Ord get_index0()const
    {
        return detail::index0_getter_<Ord,Ind,detail::template_indexer<Ord,Ind,Dims...>::value != dyn_dim,Dims...>::get(dyn_indexes0_);
    }
#endif
    Ord                total_size()const
    {
        return detail::size_calculator<Ord,Dims...>::get(dyn_dims_);
    }
};

}
}

#endif
