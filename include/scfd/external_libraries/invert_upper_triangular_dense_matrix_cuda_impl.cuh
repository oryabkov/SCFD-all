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

#ifndef __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_CUDA_IMPL_CUH__
#define __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_CUDA_IMPL_CUH__

#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include "regularize_qr_of_defect_matrix_cuda.h"

namespace scfd
{

namespace detail
{

template<class T>
__global__ void ker_r_inv_matrix_set_ident(
    int n, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < n*n)) return;
    int i = ind%n,
        j = ind/n;
    r_inv_mat[i+j*n] = (i == j? T(1) : T(0));
}

template<class T>
__global__ void ker_precalc_invert_r_diag(
    int n, T *mat_tmp, T *r_diag_inv
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < n)) return;
    r_diag_inv[i] = T(1)/mat_tmp[i*n+i];
}

template<class T>
__global__ void ker_invert_r_diag(
    int n, const T *r_diag_inv, T *mat_tmp, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < n*n)) return;
    int i = ind%n,
        j = ind/n;
    mat_tmp[i+j*n] *= r_diag_inv[i];
    if (i == j)
    {
        r_inv_mat[i+j*n] *= r_diag_inv[i];
    }
}

template<class T>
__global__ void ker_copy_mat_tmp_col_i(
    int n, int i, const T *mat_tmp, T *mat_tmp_col_i
)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i1 < i)) return;
    mat_tmp_col_i[i1] = mat_tmp[i1+i*n];
}

template<class T>
__global__ void ker_back_elimination(
    int n, int i, const T *mat_tmp_col_i, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < (n-i)*i)) return;
    int i1 = ind%i,
        j = i + ind/i;

    //make mat_tmp(i1,i) to be 0
    T mul = mat_tmp_col_i[i1];
    r_inv_mat[i1+j*n] -= r_inv_mat[i+j*n]*mul;
}

} // namespace detail

template<class T>
void invert_upper_triangular_dense_matrix_cuda(int sz, T* r_mat, T* r_diag_inv, T *mat_tmp_col_i, T* r_inv_mat)
{
    /*thrust::device_vector<thrust::pair<T,int>>  diag_with_indices(sz);
    detail::ker_get_matrix_diag_with_indices<<<(sz/256)+1,256>>>(sz, A, diag_with_indices.data().get());
    thrust::sort(diag_with_indices.begin(),diag_with_indices.end());
    thrust::device_vector<bool>                 diag_degenerate_flags(sz, false);
    detail::ker_fill_diag_degenerate_flags<<<(defect/256)+1,256>>>(
        defect, diag_with_indices.data().get(), diag_degenerate_flags.data().get()
    );
    dim3 dim_gr((sz/16)+1,(sz/16)+1),dim_bl(16,16);
    detail::ker_r_matrix_correct_defect_part<<<dim_gr,dim_bl>>>(sz, diag_degenerate_flags.data().get(), A);*/

    int block_size = 256;
    /// Init r_inv with ident matrix
    detail::ker_r_inv_matrix_set_ident<<<((sz*sz)/block_size)+1,block_size>>>(sz, r_inv_mat);
    detail::ker_precalc_invert_r_diag<<<(sz/block_size)+1,block_size>>>(sz, r_mat, r_diag_inv);
    detail::ker_invert_r_diag<<<((sz*sz)/block_size)+1,block_size>>>(sz, r_diag_inv, r_mat, r_inv_mat);

    for (int i = sz-1;i >= 0;--i)
    {
        detail::ker_copy_mat_tmp_col_i<<<(i/block_size)+1,block_size>>>(sz, i, r_mat, mat_tmp_col_i);
        ///sz([i,sz)x[0,i]) = (sz-i)*i
        detail::ker_back_elimination<<<(((sz-i)*i)/block_size)+1,block_size>>>(sz, i, mat_tmp_col_i, r_inv_mat);
        /*std::cout << "i = " << i << std::endl;
        mat_tmp_col_i.sync_from_array();
        for (int ii = 0;ii < i;++ii)
        {
            std::cout << mat_tmp_col_i(ii) << std::endl;
        }*/
        /*r_inv_mat.sync_from_array();
        for (int ii = 0;ii < sz;++ii)
        {
            for (int jj = 0;jj < sz;++jj)
            {
                std::cout << r_inv_mat(ii,jj) << " ";
            }
            std::cout << std::endl;
        }*/
    }
}

} // namespace scfd

#endif
