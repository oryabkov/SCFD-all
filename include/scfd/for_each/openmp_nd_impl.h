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

#ifndef __SCFD_FOR_EACH_OPENMP_ND_IMPL_H__
#define __SCFD_FOR_EACH_OPENMP_ND_IMPL_H__

//for_each_nd implementation for OPENMP case

#include "for_each_config.h"
#include <omp.h>
#include "openmp_nd.h"

namespace scfd
{
namespace for_each 
{

template<int dim, class T>
template<class FUNC_T>
void openmp_nd<dim,T>::operator()(FUNC_T f, const rect<T, dim> &range)const
{
    int total_sz = 1;
    for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);
    
    int real_threads_num = threads_num; //NOLINT
    if (threads_num < 0) real_threads_num = omp_get_max_threads();

    #pragma omp parallel for num_threads(real_threads_num)
    for (int i = 0;i < total_sz;++i) {
        //printf("%d %d \n", omp_get_thread_num(), omp_get_thread_num());
        vec<T, dim> idx;
        int       i_tmp = i;
        for (int j = dim-1;j >= 0;--j) {
            idx[j] = range.i1[j] + i_tmp%(range.i2[j]-range.i1[j]);
            i_tmp /= (range.i2[j]-range.i1[j]);
        }
        f(idx);
    }
}

template<int dim, class T>
template<class FUNC_T>
void openmp_nd<dim,T>::operator()(FUNC_T f, const vec<T, dim> &size)const
{
    this->operator()(f,rect<T, dim>(vec<T, dim>::make_zero(),size));
}
template<int dim, class T>
void openmp_nd<dim,T>::wait()const
{
    //void function to sync with cuda
}

}
}

#endif
