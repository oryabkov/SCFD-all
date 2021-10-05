// Copyright © 2016-2021 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_REDUCE_SERIAL_CPU_H__
#define __SCFD_REDUCE_SERIAL_CPU_H__

///TODO this is PLUS only operation reduce

#include "reduce_config.h"

namespace scfd
{
namespace reduce
{

template<class Ord = int>
struct serial_cpu
{
    /*void set_max_size(Ord max_size)
    {
        max_size_ = max_size;
    }*/

    template<class T>
    T operator()(Ord size, const T *input, T init_val)const
    {
        T   res(init_val);
        for (Ord i = 0;i < size;++i) 
        {
            res += input[i];
        }
        return res;
    }
    void    wait()const
    {
    }

    Ord     max_size_;
};

}
}

#endif
