// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_SYCL_REDUCE_H__
#define __SCFD_SYCL_REDUCE_H__

#include "reduce_config.h"

///TODO this is PLUS only operation reduce

namespace scfd
{

template<class Ord = int>
struct sycl_reduce
{
    template<class T>
    T operator()(Ord size, const T *input, T init_val)const;
    void    wait()const
    {
    }
};

}

#endif
