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

#ifndef __SCFD_ARRAYS_OBJECTS_ARRAY_H__
#define __SCFD_ARRAYS_OBJECTS_ARRAY_H__

#include "arrays_config.h"
#include <scfd/utils/device_tag.h>
#include "tensorN_array.h"

/**
* objects_array is experimental prototype aimed for
* 'large' objects T, which size is too big for single
* cuda read operation, which makes it inefficent to 
* use normal arrays
* Main interface difference with normal arrays is absence
* operator() that returns reference to the object, because
* there could be no such reference in relality - T could be
* shred in parts across the array. Instead there are get and
* set methods, that read and write T as a whole and perform
* some 'gathering' proceudres if needed
*/

namespace scfd
{
namespace arrays
{

template<class T, class Memory, 
         template <ordinal_type... Dims> class Arranger>
class objects_array_view;

template<class T, class Memory, 
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
class objects_array
{
    typedef unsigned int                                storage_t;
    static_assert(sizeof(T)%sizeof(storage_t) == 0, "objects_array: sizeof(storage_t) is not multiplier of sizeof(T)");
    typedef struct 
    {
        storage_t   d[sizeof(T)/sizeof(storage_t)];
    }                                                   value_ref_t;
    typedef tensor1_array<storage_t,Memory,
                          sizeof(T)/sizeof(storage_t),
                          Arranger>                     storage_arr_t;
public:
    typedef T                                       value_type;
    typedef Memory                                  memory_type;
    typedef objects_array_view<T,Memory,Arranger>   view_type;

public:
    void init(ordinal_type sz)
    {
        storage_arr_.init(sz);
    }
    void free()
    {
        storage_arr_.free();
    }

    bool is_free()const
    {
        return storage_arr_.is_free();
    }

    view_type           create_view(bool sync_from_array_ = true)const;

    /// obj is a first arg to be consistent with get_vec method of array
    __DEVICE_TAG__ void get(T &obj, ordinal_type i)const
    {
        value_ref_t &obj_ref = reinterpret_cast<value_ref_t&>(obj);
        #pragma unroll
        for (ordinal_type ii = 0;ii < sizeof(T)/sizeof(storage_t);++ii)
        {
            obj_ref.d[ii] = storage_arr_(i,ii);
        }
    }
    __DEVICE_TAG__ void set(const T &obj, ordinal_type i)const
    {
        const value_ref_t &obj_ref = reinterpret_cast<const value_ref_t&>(obj);
        #pragma unroll
        for (ordinal_type ii = 0;ii < sizeof(T)/sizeof(storage_t);++ii)
        {
            storage_arr_(i,ii) = obj_ref.d[ii];
        }
    }

    /// ISSUE needed for view, but think it's a bad idea. mb, use friend instead?
    __DEVICE_TAG__ const storage_arr_t   &storage_arr()const { return storage_arr_; }

private:
    storage_arr_t   storage_arr_;

};

}
}

#include "objects_array_view.h"

#endif
