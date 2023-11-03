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

#ifndef __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_CUDA_H__
#define __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_CUDA_H__

namespace scfd
{

/// r_mat is upper triangular dense square szXsz matrix in col-major format 
/// which will be rewritten by temporal data.
/// r_mat lower part is irrelevant and also would be ruined
/// r_inv_mat is matrix of the same size that will contain inv(r_mat)
/// r_diag_inv, mat_tmp_col_i are temporal buffers for vectors of size sz
template<class T>
void invert_upper_triangular_dense_matrix_cuda(int sz, T* r_mat, T* r_diag_inv, T *mat_tmp_col_i, T* r_inv_mat);

} // namespace scfd

#endif
