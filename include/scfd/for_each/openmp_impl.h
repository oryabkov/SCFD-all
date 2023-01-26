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

#ifndef __SCFD_FOR_EACH_OPENMP_IMPL_H__
#define __SCFD_FOR_EACH_OPENMP_IMPL_H__

//for_each implementation for OPENMP case

#include "for_each_config.h"
#include <omp.h>
#include "openmp.h"

namespace scfd
{
namespace for_each 
{

template<class T>
template<class FUNC_T>
void openmp<T>::operator()(FUNC_T f, T i1, T i2)const
{   
    int real_threads_num = threads_num;
    if (threads_num < 0) real_threads_num = omp_get_max_threads();

    #pragma omp parallel for num_threads(real_threads_num)
    for (T i = i1;i < i2;++i) 
    {
        f(i);
    }
}

template<class T>
template<class FUNC_T>
void openmp<T>::operator()(FUNC_T f, T size)const
{
    this->operator()(f, 0, size);
}

template<class T>
void openmp<T>::wait()const
{
    //void function to sync with cuda
}

}
}

#endif
