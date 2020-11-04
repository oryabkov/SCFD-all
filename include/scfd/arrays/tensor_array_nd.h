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

#ifndef __SCFD_ARRAYS_TENSOR_ARRAY_ND_H__
#define __SCFD_ARRAYS_TENSOR_ARRAY_ND_H__

#include <type_traits>
#include <utility>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include "tensor_base.h"
#include "detail/index_sequence.h"
#include "detail/tensor_base_nd_gen.h"
#include "detail/has_subscript_operator.h"
#include "detail/default_arranger_chooser.h"

namespace scfd
{
namespace arrays
{

using static_vec::vec;

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... Dims> class Arranger, 
         ordinal_type... TensorDims>
class tensor_array_nd_view;

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... Dims> class Arranger, 
         ordinal_type... TensorDims>
class tensor_array_nd : public detail::tensor_base_nd_gen<T,ND,Memory,Arranger,TensorDims...>::type
{
    typedef typename detail::tensor_base_nd_gen<T,ND,Memory,Arranger,TensorDims...>::type parent_t;
public:
    typedef typename parent_t::ordinal_type                                 ordinal_type;
    template<ordinal_type>using                                             index_type=ordinal_type;
    typedef tensor_array_nd_view<T,ND,Memory,Arranger,TensorDims...>        view_type;
    typedef typename parent_t::pointer_type                                 pointer_type;
    typedef typename parent_t::arranger_type                                arranger_type;

private:
    template<class IndexVec,ordinal_type... I>
    __DEVICE_TAG__ T                &index_get_(const IndexVec &idx, index_type<TensorDims>... tensor_indexes, 
                                                detail::index_sequence<ordinal_type,I...>)const
    {
        return operator()(idx[I]...,tensor_indexes...);
    }

    template<class... Args>
    struct vec_1_dim_
    {
        static const ordinal_type placeholder_ind = detail::template_arg_search<ordinal_type, placeholder, Args...>::value;
        static const ordinal_type value = detail::template_indexer<ordinal_type,placeholder_ind,TensorDims...>::value;
    };
    template<class... Args>
    struct vec_1_t_
    {
        typedef vec<T,vec_1_dim_<Args...>::value> type;
    };
    template<class... Args>
    struct vec_2_dim_
    {
        static const ordinal_type placeholder_ind = sizeof...(TensorDims)-1;
        static const ordinal_type value = detail::template_indexer<ordinal_type,placeholder_ind,TensorDims...>::value;
    };
    template<class... Args>
    struct vec_2_t_
    {
        typedef vec<T,vec_2_dim_<Args...>::value> type;
    };

public:
    __DEVICE_TAG__                  tensor_array_nd() = default;
    __DEVICE_TAG__                  tensor_array_nd(const tensor_array_nd &) = default;
    __DEVICE_TAG__                  tensor_array_nd(tensor_array_nd &&t) = default;

    __DEVICE_TAG__ tensor_array_nd  &operator=(const tensor_array_nd &t) = default;
    __DEVICE_TAG__ tensor_array_nd  &operator=(tensor_array_nd &&t) = default;

    using                           parent_t::init;
    template<class SizeVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<SizeVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args)+ND==arranger_type::dynamic_dims_num
                              >::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< std::is_integral<Args>::value... >::value
                              >::type>
    void                            init(const SizeVec &sz, Args ...args)
    {
        vec<ordinal_type,sizeof...(Args)>                      args_vec{args...};
        vec<ordinal_type,arranger_type::dynamic_dims_num>      dims_vec;
        for (int j = 0;j < ND;++j) {
            dims_vec[j] = sz[j];
        }
        for (int j = 0;j < arranger_type::dynamic_dims_num-ND;++j) {
            dims_vec[ND+j] = args_vec[j];
        }
        parent_t::init1_(dims_vec);
    }
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
    template<class SizeVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<SizeVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  (sizeof...(Args) >= (arranger_type::dynamic_dims_num-ND))&&
                                  (sizeof...(Args) <= (arranger_type::dynamic_dims_num-ND)*2)
                              >::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< std::is_integral<Args>::value... >::value
                              >::type>
    void                            init(const SizeVec &sz, const SizeVec &index0, Args... args)
    {
        vec<ordinal_type,sizeof...(Args)>                      args_vec{args...};
        vec<ordinal_type,arranger_type::dynamic_dims_num>      dims_vec, indexes0_vec;
        for (int j = 0;j < ND;++j) {
            dims_vec[j] = sz[j];
            indexes0_vec[j] = index0[j];
        }
        for (int j = 0;j < arranger_type::dynamic_dims_num-ND;++j) {
            dims_vec[ND+j] = args_vec[j];
            if (arranger_type::dynamic_dims_num-ND+j < sizeof...(Args))
                indexes0_vec[ND+j] = args_vec[arranger_type::dynamic_dims_num-ND+j];
            else
                indexes0_vec[ND+j] = 0;
        }
        parent_t::init1_(dims_vec);
        arranger_type::set_dyn_indexes0(indexes0_vec);
    }
#endif

    __DEVICE_TAG__ ordinal_type             size()const 
    {
        ordinal_type    res = 1;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < ND;++i1) res *= arranger_type::dyn_dims_[i1];
        return res;
    }
    __DEVICE_TAG__ vec<ordinal_type,ND>     size_nd()const 
    {
        vec<ordinal_type,ND>    res;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < ND;++i1) res[i1] = arranger_type::dyn_dims_[i1];
        return res;
    }

    pointer_type                    raw_ptr()const { return this->d_; }

    view_type                       create_view(bool sync_from_array_ = true)const;

    using                           parent_t::operator();
    using                           parent_t::get_vec;
    using                           parent_t::set_vec;
    template<class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<sizeof...(Args)==sizeof...(TensorDims)>::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< std::is_integral<Args>::value... >::value
                              >::type>
    __DEVICE_TAG__ T                &operator()(const IndexVec &idx, Args... args)const
    {
        return index_get_(idx, args..., detail::make_index_sequence<ordinal_type,ND>{});
    }

    template<class Vec, class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< 
                                      std::is_same<Args,placeholder>::value... 
                                  >::value == 1
                              >::type>
    __DEVICE_TAG__ void                             get_vec(Vec &v, const IndexVec &idx, Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::get_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            v[i1] = index_get_(idx, detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)..., 
                               detail::make_index_sequence<ordinal_type,ND>{});
        }
    }
    template<class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< 
                                      std::is_same<Args,placeholder>::value... 
                                  >::value == 1
                              >::type>
    __DEVICE_TAG__ typename vec_1_t_<Args...>::type get_vec(const IndexVec &idx, Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::get_vec: trying to use with dynamic index");
        typename vec_1_t_<Args...>::type     v;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            v[i1] = index_get_(idx, detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)..., 
                               detail::make_index_sequence<ordinal_type,ND>{});
        }
        return v;
    }
    template<class Vec, class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type>
    __DEVICE_TAG__ void                             get_vec(Vec &v, const IndexVec &idx, Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::get_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            v[i1] = index_get_(idx, args..., i1, 
                               detail::make_index_sequence<ordinal_type,ND>{});
        }
    }
    template<class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type>
    __DEVICE_TAG__ typename vec_2_t_<Args...>::type get_vec(const IndexVec &idx, Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::get_vec: trying to use with dynamic index");
        typename vec_2_t_<Args...>::type     v;
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            v[i1] = index_get_(idx, args..., i1, 
                               detail::make_index_sequence<ordinal_type,ND>{});
        }
        return v;
    }

    template<class Vec, class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< 
                                      std::is_same<Args,placeholder>::value... 
                                  >::value == 1
                              >::type>
    __DEVICE_TAG__ void                             set_vec(Vec &v, const IndexVec &idx, Args... args)const
    {
        static_assert(vec_1_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::set_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_1_dim_<Args...>::value;++i1) {
            index_get_(idx, detail::placeholder_get_helper<ordinal_type,Args>::get(args, i1)..., 
                       detail::make_index_sequence<ordinal_type,ND>{}) = v[i1];
        }
    }
    template<class Vec, class IndexVec, class... Args,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<Vec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  detail::has_subscript_operator<IndexVec,ordinal_type>::value
                              >::type,
             class = typename std::enable_if<
                                  sizeof...(Args) == sizeof...(TensorDims)-1
                              >::type,
             class = typename std::enable_if<
                                  detail::true_counter< std::is_integral<Args>::value... >::value == 
                                  sizeof...(TensorDims)-1
                              >::type>
    __DEVICE_TAG__ void                             set_vec(Vec &v, const IndexVec &idx, Args... args)const
    {
        static_assert(vec_2_dim_<Args...>::value != dyn_dim, 
                      "tensor_array_nd::set_vec: trying to use with dynamic index");
        #pragma unroll
        for (ordinal_type i1 = 0;i1 < vec_2_dim_<Args...>::value;++i1) {
            index_get_(idx, args..., i1, detail::make_index_sequence<ordinal_type,ND>{}) = v[i1];
        }
    }
};

}
}

#include "tensor_array_nd_view.h"

#endif