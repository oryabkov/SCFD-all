// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Ivan Antonovich Sorokin

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

#ifndef __SCFD_FOR_EACH_HIP_ND_H__
#define __SCFD_FOR_EACH_HIP_ND_H__

//for_each_nd implementation for HIP case

#include "for_each_config.h"
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>

namespace scfd
{
namespace for_each
{

using scfd::static_vec::vec;
using scfd::static_vec::rect;

template<int dim, class T = int>
struct hip_nd
{
    int block_size;

    hip_nd() : block_size(256) {}

    template<class FUNC_T>
    void operator()(const FUNC_T &f, const rect<T, dim> &range)const;
    template<class FUNC_T>
    void operator()(const FUNC_T &f, const vec<T, dim> &size)const;
    void wait()const;
};

}
}

#endif
