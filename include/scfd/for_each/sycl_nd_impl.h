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

#ifndef __SCFD_FOR_EACH_SYCL_ND_IMPL_H__
#define __SCFD_FOR_EACH_SYCL_ND_IMPL_H__

//for_each_nd implementation for OPENMP case

#include "for_each_config.h"
#include <sycl/sycl.hpp>
#include "sycl_nd.h"

namespace scfd
{
namespace for_each 
{

template<int dim, class T>
template<class FUNC_T>
void sycl_nd<dim,T>::operator()(FUNC_T f, const rect<T, dim> &range)const
{
    std::size_t total_sz = 1;
    for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);
    
    // FUNC_T must satisfy device copyable 
    // For more detail see spec, 3.13.1. Device copyable
    // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
    sycl_device_queue.parallel_for(sycl::range{total_sz}, [=](sycl::id<1> id)
    {
        //printf("%d %d \n", omp_get_thread_num(), omp_get_thread_num());
        vec<T, dim> idx;
        T i_tmp = id[0];
        for (int j = dim-1;j >= 0;--j) {
            idx[j] = range.i1[j] + i_tmp%(range.i2[j]-range.i1[j]);
            i_tmp /= (range.i2[j]-range.i1[j]);
        }
        f(idx);
    }).wait(); // assumes iterative execution model
}

template<int dim, class T>
template<class FUNC_T>
void sycl_nd<dim,T>::operator()(FUNC_T f, const vec<T, dim> &size)const
{
    this->operator()(f,rect<T, dim>(vec<T, dim>::make_zero(),size));
}
template<int dim, class T>
void sycl_nd<dim,T>::wait()const
{
    //void function to sync with cuda
}

}
}

#endif
