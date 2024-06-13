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

#ifndef __SCFD_ARRAYS_ARRAY_THRUST_CAST_H__
#define __SCFD_ARRAYS_ARRAY_THRUST_CAST_H__

#include <scfd/memory/thrust_ptr.h>

namespace scfd
{
namespace arrays
{

namespace detail {

template<class Array>
class array_thrust_ptr
{
    using value_t = typename Array::value_type;
    using memory_t = typename Array::memory_type;
    using memory_thrust_ptr_t = scfd::memory::thrust_ptr<memory_t,value_t>;
public:
    using type = typename memory_thrust_ptr_t::type;
    static type cast(value_t *p)
    {
        return memory_thrust_ptr_t::cast(p);
    }
};

} /// namespace detail

template<class Array>
typename array_thrust_ptr<Array>::type
array_thrust_begin(const Array &array)
{
    return array_thrust_ptr<Array>::cast(array.raw_ptr());
}

template<class Array>
typename array_thrust_ptr<Array>::type
array_thrust_end(const Array &array)
{
    return array_thrust_ptr<Array>::cast(array.raw_ptr()) + array.total_size();
}

} /// namespace arrays
} /// namespace scfd

#endif

