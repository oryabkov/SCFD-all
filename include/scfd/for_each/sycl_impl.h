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

#ifndef __SCFD_FOR_EACH_SYCL_IMPL_H__
#define __SCFD_FOR_EACH_SYCL_IMPL_H__

//for_each implementation for SYCL case

#include "for_each_config.h"
#include <sycl/sycl.hpp>
#include "sycl.h"

namespace scfd
{
namespace for_each
{

template<class T>
template<class FUNC_T>
void sycl_impl<T>::operator()(FUNC_T f, T i1, T i2)const
{
    std::size_t const size   = i2 - i1;
    T           const offset =      i1;

    sycl_device_queue.parallel_for(sycl::range{size}, [=](sycl::id<1> idx)
    {
        T i =  idx[0];
        f(i + offset);
    }).wait(); // assumes iterative execution model
}

template<class T>
template<class FUNC_T>
void sycl_impl<T>::operator()(FUNC_T f, T size)const
{
    this->operator()(f, 0, size);
}

template<class T>
void sycl_impl<T>::wait()const
{
}

}
}

#endif

