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


#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__VECTOR_OPERATIONS_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__VECTOR_OPERATIONS_H__


#include <array_utils.h>
#include <scfd/communication/mpi_comm_info.h>


template<class T, class Vec>
struct vector_operations
{
    using vector_type = Vec;
    
    vector_operations(scfd::communication::mpi_comm_info* mpi): mpi_(mpi)
    {}

    T all_reduce_sum(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::sum_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        return mpi_->all_reduce_sum<T>( res_local );
    }

    T all_reduce_max(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::max_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        return mpi_->all_reduce_max<T>( res_local );

    }

    T all_reduce_min(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::min_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        return mpi_->all_reduce_min<T>( res_local );
    }  

    scfd::communication::mpi_comm_info* mpi_;

};


#endif