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
//
#ifndef __SCFD_BACKEND_CUDA_H__
#define __SCFD_BACKEND_CUDA_H__

#include <scfd/backend/cuda_common.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/external_libraries/cusolver_wrap.h>

namespace scfd
{
namespace backend
{

struct cuda : public cuda_common
{
    using runtime_type = cuda;

    template <class Log>
    static int init_device( Log &log, int device_id = 0 )
    {
        return scfd::utils::init_cuda( log, -2, device_id );
    }

    static int init_device( int device_id = 0 )
    {
        return scfd::utils::init_cuda( -2, device_id );
    }

    using blas_wrap_type   = scfd::cublas_wrap;
    using solver_wrap_type = scfd::cusolver_wrap;
};

}
}


#endif // __SCFD_BACKEND_CUDA_H__
