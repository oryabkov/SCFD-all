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

#ifndef __SCFD_REGULARIZE_QR_R_SYSTEM_RHS_OF_DEFECT_MATRIX_SYSTEM_CUDA_H__
#define __SCFD_REGULARIZE_QR_R_SYSTEM_RHS_OF_DEFECT_MATRIX_SYSTEM_CUDA_H__

namespace scfd
{

/// diag_degenerate_flags must be results of regularize_qr_of_defect_matrix_cuda call.
/// r_rhs is intermediate RHS of R part of QR matrix decomposition;
/// it emerges after Q^T matrix application for the RHS of initial linear problem
template<class T>
void regularize_qr_r_system_rhs_of_defect_matrix_system_cuda(int sz, const bool *diag_degenerate_flags, T* r_rhs);

} // namespace scfd

#endif
