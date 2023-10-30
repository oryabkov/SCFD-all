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

#ifndef __SCFD_REGULARIZE_QR_OF_DEFECT_MATRIX_CUDA_H__
#define __SCFD_REGULARIZE_QR_OF_DEFECT_MATRIX_CUDA_H__

#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include "regularize_qr_of_defect_matrix_cuda.h"

namespace scfd
{

/// A must be results of cusolver.geqrf call with square A passed as system matrix
template<class T>
void regularize_qr_of_defect_matrix_cuda(int sz, T* A, int defect);

} // namespace scfd

#endif
