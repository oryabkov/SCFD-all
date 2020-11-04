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

#ifndef __SCFD_ARRAYS_TENSOR_ARRAY_ND_VISIBLE_H__
#define __SCFD_ARRAYS_TENSOR_ARRAY_ND_VISIBLE_H__

#include <type_traits>
#include <utility>
#include "tensor_array_nd.h"
#include "detail/default_arranger_chooser.h"

namespace scfd
{
namespace arrays
{

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... Dims> class Arranger, 
         ordinal_type... TensorDims>
class tensor_array_nd_visible : public tensor_array_nd<T, ND, typename Memory::host_memory_type, Arranger, TensorDims...>
{
    typedef tensor_array_nd<T, ND, typename Memory::host_memory_type, Arranger, TensorDims...> parent_t;
public: 
    typedef tensor_array_nd<T,ND,Memory,Arranger,TensorDims...>         array_type;
    typedef typename parent_t::arranger_type                            arranger_type;
    typedef Memory                                                      array_memory_type;
    static const bool                                                   has_separate_buffer = !Memory::is_host_visible;
private:

    array_type  array_;

    template<bool has_separate_buffer_>
    typename std::enable_if<has_separate_buffer_>::type     
    init_by_(const array_type &array)
    {
        parent_t::init_alike(array);
    }
    template<bool has_separate_buffer_>
    typename std::enable_if<!has_separate_buffer_>::type    
    init_by_(const array_type &array)
    {
        parent_t::assign(array);
    }
    //just to shadow init_alike from parent
    void                            init_alike() { }
public:
    tensor_array_nd_visible() = default;
    tensor_array_nd_visible(const tensor_array_nd_visible &) = delete;
    tensor_array_nd_visible(tensor_array_nd_visible &&) = default;

    tensor_array_nd_visible         &operator=(const tensor_array_nd_visible &) = delete;
    tensor_array_nd_visible         &operator=(tensor_array_nd_visible &&) = default;

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
        array_.init(args...);
        init_by_<has_separate_buffer>(array_);
    }
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
        array_.init(sz, args...);
        init_by_<has_separate_buffer>(array_);
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
        array_.init(sz, index0, args...);
        init_by_<has_separate_buffer>(array_);
    }
#endif
    void                            free()
    {
        if (has_separate_buffer) parent_t::free();
        array_.free();
    }

    array_type                      array()const { return array_; }

    void                            sync_to_array()const
    {
        if (!has_separate_buffer) return;
        array_memory_type::copy_from_host(sizeof(T)*arranger_type::total_size(), this->raw_ptr(), array_.raw_ptr());
    }
    void                            sync_from_array()const
    {
        if (!has_separate_buffer) return;
        array_memory_type::copy_to_host(sizeof(T)*arranger_type::total_size(), array_.raw_ptr(), this->raw_ptr());
    }
};

}
}

#endif
