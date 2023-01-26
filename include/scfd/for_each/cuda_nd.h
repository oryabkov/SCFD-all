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

#ifndef __SCFD_FOR_EACH_CUDA_ND_H__
#define __SCFD_FOR_EACH_CUDA_ND_H__

//for_each_nd implementation for CUDA case

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
struct cuda_nd
{
    //rect<T, dim> block_size;    //ISSUE remove from here somehow: we need it just 4 cuda
    int block_size;

    template<class FUNC_T>
    void operator()(const FUNC_T &f, const rect<T, dim> &range)const;
    template<class FUNC_T>
    void operator()(const FUNC_T &f, const vec<T, dim> &size)const;
    void wait()const;
};

}
}

#endif
