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

#ifndef __SCFD_BACKEND_SERIAL_CPU_MPI_H__
#define __SCFD_BACKEND_SERIAL_CPU_MPI_H__

#include <type_traits>
#include <scfd/backend/serial_cpu.h>
#include <scfd/communication/mpi_comm_info.h>

namespace scfd
{
namespace backend
{

struct serial_cpu_mpi : public serial_cpu
{
    using serial_cpu::init_device;
    using communicator_type = scfd::communication::mpi_comm_info;
    using runtime_type      = serial_cpu_mpi;

    template <
        class Log, class Comm,
        typename std::enable_if<!std::is_integral<typename std::decay<Comm>::type>::value, int>::type = 0>
    static int init_device( Log &, const Comm &, int = 0, bool = false )
    {
        return 0;
    }

    template <class Comm>
    static int init_device( const Comm &, int = 0, bool = false )
    {
        return 0;
    }
};

}
}

#endif
