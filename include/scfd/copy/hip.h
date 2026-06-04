// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_HIP_COPY_H__
#define __SCFD_HIP_COPY_H__

#include <hip/hip_runtime.h>
#include <scfd/utils/hip_safe_call.h>

namespace scfd
{

template <class Ord = int>
struct hip_copy
{
    template <class T>
    void operator()( Ord size, const T *input, T *output ) const
    {
        if ( size <= 0 )
            return;
        HIP_SAFE_CALL( hipMemcpy( output, input, sizeof( T ) * static_cast<size_t>( size ), hipMemcpyDefault ) );
    }
    void wait() const
    {
    }
};

}

#endif
