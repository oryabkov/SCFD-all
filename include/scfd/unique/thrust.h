// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UNIQUE_THRUST_H__
#define __SCFD_UNIQUE_THRUST_H__

#include "unique_config.h"
#include <thrust/device_ptr.h>
#include <thrust/unique.h>

namespace scfd
{

template <class Ord = int>
struct thrust_unique
{
    template <class T>
    Ord operator()( Ord size, T *data ) const
    {
        ::thrust::device_ptr<T> data_begin = ::thrust::device_pointer_cast( data ),
                                data_end   = data_begin + size;
        auto end = ::thrust::unique( data_begin, data_end );
        return static_cast<Ord>( end - data_begin );
    }
    void wait() const
    {
    }
};

}

#endif
