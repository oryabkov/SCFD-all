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

#ifndef __SCFD_ARRAYS_TENSOR_BASE_H__
#define __SCFD_ARRAYS_TENSOR_BASE_H__

#include "arrays_config.h"
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include "placeholder.h"
#include "detail/template_indexer.h"
#include "detail/index_sequence.h"
#include "detail/bool_array.h"
#include "detail/true_counter.h"
#include "detail/placeholder_get_helper.h"
#include "detail/template_arg_search.h"
#include "detail/has_subscript_operator.h"

namespace scfd
{
namespace arrays
{

using static_vec::vec;

template<class T, class Memory, 
         template <ordinal_type... DimsA> class Arranger, 
         ordinal_type... Dims>
class tensor_base : public Arranger<Dims...>
{
public:
    typedef T                       value_type;
    typedef T*                      pointer_type;
    typedef arrays::ordinal_type    ordinal_type;
    typedef Memory                  memory_type;
    typedef Arranger<Dims...>       arranger_type;

protected:
    pointer_type    d_;
    //own_ means whether we should free d_-pointed memory when this object dies
    //own_ has meaning if and only if array is not free (is_free() == false)
    bool            own_;

protected:
    void                            init1_(const vec<ordinal_type,arranger_type::dynamic_dims_num> &dyn_dims) 
    {
        assert(is_free());
        arranger_type::set_dyn_dims(dyn_dims);
        memory_type::malloc((void**)&d_, sizeof(T)*arranger_type::total_size());
        own_ = true;
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        arranger_type::set_zero_dyn_indexes0();
#endif
        assert((!is_free() || (arranger_type::total_size() == 0)) && own_);
    }
    void                            init1_by_raw_data_(pointer_type raw_data_ptr, 
                                                       const vec<ordinal_type,arranger_type::dynamic_dims_num> &dyn_dims)
    {
        assert(is_free());
        arranger_type::set_dyn_dims(dyn_dims);
        if (arranger_type::total_size() != 0)
            d_ = raw_data_ptr;
        else
            d_ = NULL;
        own_ = false;
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        arranger_type::set_zero_dyn_indexes0();
#endif
        assert((!is_free() || (arranger_type::total_size() == 0)) && !own_);
    }

    __DEVICE_TAG__ void             assign(const tensor_base &t) 
    { 
        arranger_type::operator=(t); 
        d_ = t.d_; own_ = false; 
    }
    __DEVICE_TAG__ void             move(tensor_base &&t) 
    { 
        arranger_type::operator=(t); 
        d_ = t.d_; own_ = t.own_; 
        t.d_ = NULL; t.own_ = false; 
    }

    template<class... Indexes,
             class = typename std::enable_if<sizeof...(Indexes)==sizeof...(Dims)>::type,
             class = typename std::enable_if<detail::check_all_are_true< std::is_integral<Indexes>::value... >::value>::type>
    __DEVICE_TAG__ T                &operator()(Indexes... indexes)const
    {
        return d_[arranger_type::calc_lin_index(indexes...)];
    }  

    template<class... Args>
    struct vec_1_dim_
    {
        static const ordinal_type placeholder_ind = detail::template_arg_search<int, placeholder, Args...>::value;
        static const ordinal_type value = detail::template_indexer<ordinal_type,placeholder_ind,Dims...>::value;
    };
    template<class... Args>
    struct vec_1_t_
    {
        typedef vec<T,vec_1_dim_<Args...>::value> type;
    };
    template<class... Args>
    struct vec_2_dim_
    {
        static const ordinal_type placeholder_ind = sizeof...(Dims)-1;
        static const ordinal_type value = detail::template_indexer<ordinal_type,placeholder_ind,Dims...>::value;
    };
    template<class... Args>
    struct vec_2_t_
    {
        typedef vec<T,vec_2_dim_<Args...>::value> type;
    };

    template<class Vec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_same<Args,placeholder>::value... >::value == 1
                              >::type>
    __DEVICE_TAG__ void                         get_vec(Vec &v, Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::get_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            v[i1] = this->operator()(detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)...);
        }
    }
    template<class... Args,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_same<Args,
                                  placeholder>::value... >::value == 1
                              >::type>
    __DEVICE_TAG__ typename vec_1_t_<Args...>::type get_vec(Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::get_vec: trying to use with dynamic index");
        typename vec_1_t_<Args...>::type     v;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            v[i1] = this->operator()(detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)...);
        }
        return v;
    }
    template<class Vec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type>
    __DEVICE_TAG__ void                             get_vec(Vec &v, Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::get_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            v[i1] = this->operator()(args..., i1);
        }
    }
    template<class... Args,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type>
    __DEVICE_TAG__ typename vec_2_t_<Args...>::type get_vec(Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::get_vec: trying to use with dynamic index");
        typename vec_2_t_<Args...>::type     v;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            v[i1] = this->operator()(args..., i1);
        }
        return v;
    }
    template<class Vec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_same<Args,placeholder>::value... >::value == 1
                              >::type>
    __DEVICE_TAG__ void                             set_vec(const Vec &v, Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::set_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            this->operator()(detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)...) = v[i1];
        }
    }
    template<class Vec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args)==sizeof...(Dims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(Dims)-1
                              >::type>
    __DEVICE_TAG__ void                             set_vec(const Vec &v, Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_base::set_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            this->operator()(args..., i1) = v[i1];
        }
    }

public:
    __DEVICE_TAG__                  tensor_base() : d_(NULL) {}
    __DEVICE_TAG__                  tensor_base(const tensor_base &t) { assign(t); }
    __DEVICE_TAG__                  tensor_base(tensor_base &&t) { move(std::move(t)); }

    __DEVICE_TAG__ tensor_base      &operator=(const tensor_base &t) { assign(t); return *this; }
    __DEVICE_TAG__ tensor_base      &operator=(tensor_base &&t) { move(std::move(t)); return *this; }

    __DEVICE_TAG__ bool             is_free()const { return d_ == NULL; }
    __DEVICE_TAG__ bool             is_own()const { return own_; }

    template<class... Args,
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
             class = typename std::enable_if<(sizeof...(Args) >= arranger_type::dynamic_dims_num)&&
                                             (sizeof...(Args) <= arranger_type::dynamic_dims_num*2)>::type,
#else
             class = typename std::enable_if<sizeof...(Args)==arranger_type::dynamic_dims_num>::type,
#endif             
             class = typename std::enable_if<detail::check_all_are_true< std::is_integral<Args>::value... >::value>::type>
    void                            init(Args ...args)
    {
        vec<ordinal_type,sizeof...(Args)>                      args_vec{args...};
        vec<ordinal_type,arranger_type::dynamic_dims_num>      dims_vec, indexes0_vec;
        for (int j = 0;j < arranger_type::dynamic_dims_num;++j) {
            dims_vec[j] = args_vec[j];
            if (arranger_type::dynamic_dims_num+j < sizeof...(Args)) { 
                indexes0_vec[j] = args_vec[arranger_type::dynamic_dims_num+j];
            } else {
                indexes0_vec[j] = 0;
            }
        }
        init1_(dims_vec);
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        arranger_type::set_dyn_indexes0(indexes0_vec);
#endif
    }
    template<class... Args,
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
             class = typename std::enable_if<(sizeof...(Args) >= arranger_type::dynamic_dims_num)&&
                                             (sizeof...(Args) <= arranger_type::dynamic_dims_num*2)>::type,
#else
             class = typename std::enable_if<sizeof...(Args)==arranger_type::dynamic_dims_num>::type,
#endif             
             class = typename std::enable_if<detail::check_all_are_true< std::is_integral<Args>::value... >::value>::type>
    void                            init_by_raw_data(pointer_type raw_data_ptr,
                                                     Args ...args)
    {
        vec<ordinal_type,sizeof...(Args)>                      args_vec{args...};
        vec<ordinal_type,arranger_type::dynamic_dims_num>      dims_vec, indexes0_vec;
        for (int j = 0;j < arranger_type::dynamic_dims_num;++j) {
            dims_vec[j] = args_vec[j];
            if (arranger_type::dynamic_dims_num+j < sizeof...(Args)) { 
                indexes0_vec[j] = args_vec[arranger_type::dynamic_dims_num+j];
            } else {
                indexes0_vec[j] = 0;
            }
        }
        init1_by_raw_data_(raw_data_ptr,dims_vec);
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        arranger_type::set_dyn_indexes0(indexes0_vec);
#endif
    }
    void                            init_alike(const arranger_type& a)
    {
        assert(is_free());
        arranger_type::copy_dyn_shape(a);
        memory_type::malloc((void**)&d_, sizeof(T)*arranger_type::total_size());
        own_ = true;
    }
    void                            free()
    {
        if (is_free()) return;
        assert(own_);
        memory_type::free(d_);
        d_ = NULL;
    }

#ifndef __CUDA_ARCH__
    ~tensor_base()
    {
        //TODO we must catch exceptions here
        if (!is_free() && own_) free();
    }
#endif

};

}
}

#endif
