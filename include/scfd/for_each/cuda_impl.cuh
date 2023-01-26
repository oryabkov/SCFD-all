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

#ifndef __SCFD_FOR_EACH_CUDA_IMPL_CUH__
#define __SCFD_FOR_EACH_CUDA_IMPL_CUH__

//for_each implementation for CUDA case

#include <cuda_runtime.h>
#include "for_each_config.h"
#include "cuda.h"

namespace scfd
{
namespace for_each 
{

template<class FUNC_T, class T>
__global__ void ker_for_each(FUNC_T f, T i1, T i2)
{
    T i = i1 + blockIdx.x*blockDim.x + threadIdx.x;
    if (!((i >= i1)&&(i < i2))) return;
    f(i);
}

template<class T>
template<class FUNC_T>
void cuda<T>::operator()(FUNC_T f, T i1, T i2)const
{
    T total_sz = i2-i1;
    ker_for_each<FUNC_T,T><<<(total_sz/block_size)+1,block_size>>>(f, i1, i2);
}

template<class T>
template<class FUNC_T>
void cuda<T>::operator()(FUNC_T f, T size)const
{
    this->operator()(f, 0, size);
}

template<class T>
void cuda<T>::wait()const
{
    //TODO error check?
    CUDA_SAFE_CALL( cudaStreamSynchronize(0) );
}

}
}

#endif
