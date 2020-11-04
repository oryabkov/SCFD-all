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

#ifndef __SCFD_ARRAYS_TENSOR_ARRAY_ND_VIEW_H__
#define __SCFD_ARRAYS_TENSOR_ARRAY_ND_VIEW_H__

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
class tensor_array_nd_view : public tensor_array_nd<T, ND, typename Memory::host_memory_type, Arranger, TensorDims...>
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
    void                                init_alike() { }
public:
    tensor_array_nd_view() = default;
    tensor_array_nd_view(const tensor_array_nd_view &) = delete;
    tensor_array_nd_view(tensor_array_nd_view &&) = default;
    tensor_array_nd_view(const array_type &array, bool sync_from_array_ = true)
    {
        init(array, sync_from_array_);
    }

    tensor_array_nd_view                &operator=(const tensor_array_nd_view &) = delete;
    tensor_array_nd_view                &operator=(tensor_array_nd_view &&) = default;

    void                                init(const array_type &array, bool sync_from_array_ = true) 
    {
        assert(this->is_free()); assert(array_.is_free());
        array_ = array;
        init_by_<has_separate_buffer>(array);
        if (sync_from_array_) sync_from_array();
    }
    void                                release(bool sync_to_array_ = true)
    {
        if (sync_to_array_) sync_to_array();
        if (has_separate_buffer) 
            parent_t::free();
        else 
            *this = array_type();
        array_ = array_type();
    }
    void                                free()
    {
        release(false);
    }

    void                                sync_to_array()const
    {
        if (!has_separate_buffer) return;
        array_memory_type::copy_from_host(sizeof(T)*arranger_type::total_size(), this->raw_ptr(), array_.raw_ptr());
    }
    void                                sync_from_array()const
    {
        if (!has_separate_buffer) return;
        array_memory_type::copy_to_host(sizeof(T)*arranger_type::total_size(), array_.raw_ptr(), this->raw_ptr());
    }
};

template<class T, ordinal_type ND, class Memory, 
         template <ordinal_type... Dims> class Arranger, 
         ordinal_type... TensorDims>
typename tensor_array_nd<T,ND,Memory,Arranger,TensorDims...>::view_type   
tensor_array_nd<T,ND,Memory,Arranger,TensorDims...>::create_view(bool sync_from_array_)const
{
    return view_type(*this, sync_from_array_);
}

}
}

#endif
