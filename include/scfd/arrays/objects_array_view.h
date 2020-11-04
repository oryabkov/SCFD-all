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

#ifndef __SCFD_ARRAYS_OBJECTS_ARRAY_VIEW_H__
#define __SCFD_ARRAYS_OBJECTS_ARRAY_VIEW_H__

#include "arrays_config.h"
#include "objects_array.h"
#include "tensorN_array_view.h"

namespace scfd
{
namespace arrays
{

template<class T, class Memory, 
         template <ordinal_type... Dims> class Arranger>
class objects_array_view
{
public:
    typedef objects_array<T,Memory,Arranger>    array_type;
    typedef T                                   value_type;
    typedef typename Memory::host_memory_type   memory_type;

public:
    objects_array_view() = default;
    objects_array_view(const objects_array_view &) = delete;
    objects_array_view(objects_array_view &&) = default;
    objects_array_view(const array_type &array, bool sync_from_array_ = true)
    {
        init(array, sync_from_array_);
    }

    void                                init(const array_type &array, bool sync_from_array_ = true) 
    {
        storage_arr_view_.init(array.storage_arr(), sync_from_array_);
    }
    void                                release(bool sync_to_array_ = true)
    {
        storage_arr_view_.release(sync_to_array_);
    }
    void                                free()
    {
        storage_arr_view_.free();
    }

    void                                sync_to_array()const
    {
        storage_arr_view_.sync_to_array();
    }
    void                                sync_from_array()const
    {
        storage_arr_view_.sync_from_array();
    }

    /// obj is a first arg to be consistent with get_vec method of array
    void get(T &obj, ordinal_type i)const
    {
        value_ref_t &obj_ref = reinterpret_cast<value_ref_t&>(obj);
        #pragma unroll
        for (ordinal_type ii = 0;ii < sizeof(T)/sizeof(storage_t);++ii)
        {
            obj_ref.d[ii] = storage_arr_view_(i,ii);
        }
    }
    void set(const T &obj, ordinal_type i)const
    {
        const value_ref_t &obj_ref = reinterpret_cast<const value_ref_t&>(obj);
        #pragma unroll
        for (ordinal_type ii = 0;ii < sizeof(T)/sizeof(storage_t);++ii)
        {
            storage_arr_view_(i,ii) = obj_ref.d[ii];
        }
    }
private:
    typedef unsigned int                                    storage_t;
    static_assert(sizeof(T)%sizeof(storage_t) == 0, "objects_array: sizeof(storage_t) is not multiplier of sizeof(T)");
    typedef struct 
    {
        storage_t   d[sizeof(T)/sizeof(storage_t)];
    }                                                       value_ref_t;
    typedef tensor1_array_view<storage_t,Memory,
                               sizeof(T)/sizeof(storage_t),
                               Arranger>                    storage_arr_view_t;

private:
    storage_arr_view_t   storage_arr_view_;

};

template<class T, class Memory, 
         template <ordinal_type... Dims> class Arranger>
typename objects_array<T,Memory,Arranger>::view_type   
objects_array<T,Memory,Arranger>::create_view(bool sync_from_array_)const
{
    return view_type(*this, sync_from_array_);
}

}
}

#endif
