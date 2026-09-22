// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_PLATFORM_CUDA_H__
#define __SCFD_PLATFORM_CUDA_H__

#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/init_cuda_mpi.h>

namespace scfd
{
namespace platform
{

template <
    class Ordinal = PLATFORM_ORDINAL, class BigOrdinal = PLATFORM_BIG_ORDINAL,
    class Communicator = scfd::communication::mpi_comm_info>
struct cuda_mpi : public backend::cuda<Ordinal>
{
    using backend_type                   = backend::cuda<Ordinal>;
    using ordinal_type                   = Ordinal;
    using big_ordinal_type               = BigOrdinal;
    using communicator_type              = Communicator;
    using communication_environment_type = scfd::communication::mpi_wrap;
    using runtime_type                   = cuda_mpi<Ordinal, BigOrdinal, Communicator>;
    using backend_type::init_device;

    template <class Log>
    static int init( Log &log, const communicator_type &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        return scfd::utils::init_cuda_mpi( log, comm, shift_index, wrap_procs_devices );
    }

    static int init( const communicator_type &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        return scfd::utils::init_cuda_mpi( comm, shift_index, wrap_procs_devices );
    }
};

}
}

#endif
