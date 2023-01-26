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

#ifndef __SCFD_FOR_EACH_CUDA_ND_IMPL_CUH__
#define __SCFD_FOR_EACH_CUDA_ND_IMPL_CUH__

//for_each_nd implementation for CUDA case

#include "for_each_config.h"
#include "cuda_nd.h"

namespace scfd
{
namespace for_each 
{

template<class FUNC_T, int dim, class T>
__global__ void ker_for_each(FUNC_T f, rect<T, dim> range, int total_sz)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (!((i >= 0)&&(i < total_sz))) return;
    vec<T, dim> idx;
    for (int j = 0;j < dim;++j) {
        idx[j] = range.i1[j] + i%(range.i2[j]-range.i1[j]);
        i /= (range.i2[j]-range.i1[j]);
    }
    f(idx);
}

template<int dim, class T>
template<class FUNC_T>
void cuda_nd<dim,T>::operator()(const FUNC_T &f, const rect<T, dim> &range)const
{
    int total_sz = 1;
    for (int j = 0;j < dim;++j) total_sz *= (range.i2[j]-range.i1[j]);
    
    ker_for_each<FUNC_T,dim,T><<<(total_sz/block_size)+1,block_size>>>(f, range, total_sz);
}

template<int dim, class T>
template<class FUNC_T>
void cuda_nd<dim,T>::operator()(const FUNC_T &f, const vec<T, dim> &size)const
{
    this->operator()(f,rect<T, dim>(vec<T, dim>::make_zero(),size));
}

template<int dim, class T>
void cuda_nd<dim,T>::wait()const
{
    //TODO error check?
    CUDA_SAFE_CALL( cudaStreamSynchronize(0) );
}

}
}

#endif
