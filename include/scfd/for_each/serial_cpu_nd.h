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

#ifndef __SCFD_FOR_EACH_SERIAL_CPU_ND_H__
#define __SCFD_FOR_EACH_SERIAL_CPU_ND_H__

#include "for_each_config.h"
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>

//ISSUE think about different interface to FUNC_T. Instead of passing idx variable may be it is better to pass some kind of iterface (which calc it on fly - to prevent extra registers pressure)

namespace scfd
{
namespace for_each 
{

using scfd::static_vec::vec;
using scfd::static_vec::rect;

//T is ordinal type (like int)
template<int dim, class T = int>
struct serial_cpu_nd
{
    inline bool next(const rect<T, dim> &range, vec<T, dim> &idx)const
    {
        for (int j = dim-1;j >= 0;--j) {
            ++(idx[j]);
            if (idx[j] < range.i2[j]) return true;
            idx[j] = range.i1[j];
        }
        return false;
    }

    //FUNC_T concept:
    //TODO
    //copy-constructable
    template<class FUNC_T>
    void operator()(FUNC_T f, const rect<T, dim> &range)const
    {
        vec<T, dim> idx = range.i1;
        do {
            f(idx);
        } while (next(range, idx));
    }
    template<class FUNC_T>
    void operator()(FUNC_T f, const vec<T, dim> &size)const
    {
        this->operator()(f,rect<T, dim>(vec<T, dim>::make_zero(),size));
    }
    void wait()const
    {
        //void function to sync with cuda
    }
};

}
}

#endif
