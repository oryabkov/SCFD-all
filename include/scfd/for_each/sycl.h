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

#ifndef __SCFD_FOR_EACH_SYCL_H_
#define __SCFD_FOR_EACH_SYCL_H__

//for_each implementation for SYCL case

#ifdef SCFD_FOR_EACH_ENABLE_PROPERTY_TREE_INIT
#include <boost/property_tree/ptree.hpp>
#endif
#include "scfd/utils/init_sycl.hpp"
#include "for_each_config.h"
#include <sycl/sycl.hpp>

namespace scfd
{
namespace for_each
{

template<class T = int>
struct sycl 
{
    sycl() : threads_num(-1) {}
    int threads_num;

    template<class FUNC_T>
    void operator()(FUNC_T f, T i1, T i2)const;
    template<class FUNC_T>
    void operator()(FUNC_T f, T size)const;
    void wait()const;

    #ifdef SCFD_FOR_EACH_ENABLE_PROPERTY_TREE_INIT
    void init(const boost::property_tree::ptree &cfg)
    {
        threads_num = cfg.get<int>("threads_num", -1);
    }
    #endif
};

}
}

#endif
