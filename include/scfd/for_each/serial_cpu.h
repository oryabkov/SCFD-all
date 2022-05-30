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

#ifndef __SCFD_FOR_EACH_SERIAL_CPU_H__
#define __SCFD_FOR_EACH_SERIAL_CPU_H__

#ifdef SCFD_FOR_EACH_ENABLE_PROPERTY_TREE_INIT
#include <boost/property_tree/ptree.hpp>
#endif
#include "for_each_config.h"

//TODO rename for_each to t_for_each_tml; do the same for for_each_ndim; copy this all to tensor_fileds_and_foreach

namespace scfd
{
namespace for_each 
{

//T is ordinal type (like int)
//SERIAL_CPU realization is default
template<class T = int>
struct serial_cpu
{
    //FUNC_T concept:
    //TODO
    //copy-constructable
    template<class FUNC_T>
    void operator()(FUNC_T f, T i1, T i2)const
    {
        for (T i = i1;i < i2;++i) 
        {
            f(i);
        }
    }
    void    wait()const
    {
    }
    #ifdef SCFD_FOR_EACH_ENABLE_PROPERTY_TREE_INIT
    void init(const boost::property_tree::ptree &cfg) { }
    #endif
};

}
}

#endif
