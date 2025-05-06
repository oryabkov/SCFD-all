// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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


#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__FUNCTORS_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__FUNCTORS_H__

#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>


namespace functors
{

namespace kernel
{



// template<class Ord, class T>
// __global__ void fill_values_ker(Ord N, Ord shift, T* array)
// {
//     Ord i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i>=N) return;
//     array[i] = static_cast<T>(shift);
// }

template<class Ord, class T, int Dim, class Array>
struct fill_values
{
    fill_values(Ord shift, Array array):
    array_(array),
    shift_(shift)
    {}
    Array array_;
    Ord shift_;

    __DEVICE_TAG__ void operator()(const scfd::static_vec::vec<Ord, Dim> &idx)
    {
        array_(idx) = static_cast<T>(shift_);
    }
};


template<class Ord, int Dim, class Array>
struct make_zero
{
    make_zero(Array& output):
    output_(output)
    {}
    Array output_;
    __DEVICE_TAG__ void operator()(const scfd::static_vec::vec<Ord, Dim> &idx)
    {
        output_(idx) = 0;
    }
};
}


template<class ForEach, class Ord, class T, int Dim, class Array>    
void fill_values( const ForEach &for_each, const scfd::static_vec::rect<Ord,Dim> &rect, Ord shift, Array array )
{   
    for_each( kernel::fill_values<Ord, T, Dim, Array>(shift, array), rect);
}


// template<class Ord, class T, int Dim, class Array>    
// void fill_values_ker(const scfd::static_vec::rect<Ord,Dim> &rect, Ord shift, Array array )
// {   
//     Ord N = rect.calc_area();
//     Ord BLOCKSIZE = 256;
//     dim3 dim_block(BLOCKSIZE);
//     Ord grid = (N+BLOCKSIZE)/BLOCKSIZE;
//     dim3 dim_grid( grid );

//     kernel::fill_values_ker<Ord, T><<<dim_grid, dim_block>>>(N, shift, array.raw_ptr() );
// }

template<class ForEach, class Ord, int Dim, class Array>    
void make_zero( const ForEach &for_each, const scfd::static_vec::rect<Ord,Dim> &rect, Array output )
{   
    for_each( kernel::make_zero<Ord, Dim, Array>(output), rect);
}



}


#endif
