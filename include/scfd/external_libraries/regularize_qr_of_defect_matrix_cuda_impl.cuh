// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_REGULARIZE_QR_OF_DEFECT_MATRIX_CUDA_IMPL_CUH__
#define __SCFD_REGULARIZE_QR_OF_DEFECT_MATRIX_CUDA_IMPL_CUH__

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include "regularize_qr_of_defect_matrix_cuda.h"

namespace scfd
{

namespace detail
{

template<class T>
__global__ void ker_get_matrix_diag_with_indices(
    int n, const T *A, thrust::pair<T,int> *diag_with_indices
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < n)) return;
    //TODO fabs->scalar_trait
    diag_with_indices[i] = thrust::make_pair<T,int>(fabs(A[i*n+i]),i);
}

template<class T>
__global__ void ker_fill_diag_degenerate_flags(
    int defect, const thrust::pair<T,int> *diag_with_indices, bool *diag_degenerate_flags
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < defect)) return;
    int degenerate_diag_index = diag_with_indices[i].second;
    diag_degenerate_flags[degenerate_diag_index] = true;
}

template<class T>
__global__ void ker_r_matrix_correct_defect_part(
    int n, const bool *diag_degenerate_flags, T *A
)
{
    /// i is row number and j is col number
    /// Here fast index x is chosen according to ColMajor matrix A layout
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (!((i < n)&&(j < n))) return;
    if (i > j) return; /// Lower part is not affected
    if (i == j) /// Diagonal threat
    {
        if (diag_degenerate_flags[i])
        {
            A[i+j*n] = T(1);
        }
    }
    else
    {
        if (diag_degenerate_flags[i]||diag_degenerate_flags[j])
        {
            A[i+j*n] = T(0);
        }
        ////TEST
        //A[i+j*n] = T(0);
    }
}

} // namespace detail

/// A must be results of cusolver.geqrf call with square szXsz A passed as system matrix
/// diag_degenerate_flags is bool vector of size sz that will contain flags for futher calls
/// to regularize_qr_r_system_rhs_of_defect_matrix_system_cuda; diag_degenerate_flags must be allocated
/// even if it is not needed as output (because function will use it internally)
template<class T>
void regularize_qr_of_defect_matrix_cuda(int sz, T* A, bool *diag_degenerate_flags, int defect)
{
    thrust::device_vector<thrust::pair<T,int>>  diag_with_indices(sz);
    detail::ker_get_matrix_diag_with_indices<<<(sz/256)+1,256>>>(sz, A, diag_with_indices.data().get());
    thrust::sort(diag_with_indices.begin(),diag_with_indices.end());
    thrust::fill(
        thrust::device_pointer_cast(diag_degenerate_flags),
        thrust::device_pointer_cast(diag_degenerate_flags) + sz,
        false
    );
    detail::ker_fill_diag_degenerate_flags<<<(defect/256)+1,256>>>(
        defect, diag_with_indices.data().get(), diag_degenerate_flags
    );
    dim3 dim_gr((sz/16)+1,(sz/16)+1),dim_bl(16,16);
    detail::ker_r_matrix_correct_defect_part<<<dim_gr,dim_bl>>>(sz, diag_degenerate_flags, A);
}

} // namespace scfd

#endif
