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

#ifndef __SCFD_EXCLUSIVE_SCAN_THRUST_H__
#define __SCFD_EXCLUSIVE_SCAN_THRUST_H__

#include "exclusive_scan_config.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace scfd
{

template <class Ord = int>
struct thrust_exclusive_scan
{
    template <class T>
    void operator()( Ord size, const T *input, T *output, T init_val ) const
    {
        ::thrust::device_ptr<const T> input_begin = ::thrust::device_pointer_cast( input ),
                                      input_end   = input_begin + size;
        ::thrust::device_ptr<T> output_begin = ::thrust::device_pointer_cast( output ),
                                 output_end   = output_begin + size;
        ::thrust::exclusive_scan( input_begin, input_end, output_begin, init_val );
    }
    void wait() const
    {
    }
};

}

#endif
