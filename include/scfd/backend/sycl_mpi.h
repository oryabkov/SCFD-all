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

#ifndef __SCFD_BACKEND_SYCL_MPI_H__
#define __SCFD_BACKEND_SYCL_MPI_H__

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <scfd/backend/sycl.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/utils/log_std.h>

namespace scfd
{
namespace backend
{

struct sycl_mpi : public sycl
{
    using sycl::init_device;
    using communicator_type = scfd::communication::mpi_comm_info;
    using runtime_type      = sycl_mpi;

    template <
        class Log, class Comm,
        typename std::enable_if<!std::is_integral<typename std::decay<Comm>::type>::value, int>::type = 0>
    static int init_device( Log &log, const Comm &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        auto node_comm = comm.split_type( MPI_COMM_TYPE_SHARED );
        int  node_size = node_comm.num_procs();
        int  my_id     = node_comm.myid();
        node_comm.free();

        std::vector<::sycl::device> devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
        const int                   number_of_devices_on_node = static_cast<int>( devices.size() );
        if ( number_of_devices_on_node <= 0 )
            throw std::runtime_error( "sycl_mpi::init_device: no visible SYCL GPU devices" );
        if ( number_of_devices_on_node < node_size && !wrap_procs_devices )
        {
            throw std::runtime_error(
                "sycl_mpi::init_device: number of nproc = " + std::to_string( node_size ) +
                ", number of SYCL GPU devices = " + std::to_string( number_of_devices_on_node ) +
                "\n numproc per node > numDevices per node"
            );
        }

        const int device_id = ( my_id + shift_index ) % number_of_devices_on_node;
        if ( number_of_devices_on_node < node_size && wrap_procs_devices && my_id == 0 )
        {
            log.info_f(
                "WARNING: sycl_mpi::init_device is wrapping %i MPI processes over %i visible device(s). "
                "Several MPI processes will share one device.",
                node_size, number_of_devices_on_node
            );
        }
        log.info_f(
            "sycl_mpi::init_device: global_size = %i, global_id = %i, node_size = %i, devices_on_node = %i, "
            "node_device_id = %i, node_my_id = %i",
            comm.num_procs, comm.myid, node_size, number_of_devices_on_node, device_id, my_id
        );
        sycl_device_queue = ::sycl::queue( devices[static_cast<std::size_t>( device_id )] );
        return device_id;
    }

    template <class Comm>
    static int init_device( const Comm &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        scfd::utils::log_std log;
        return init_device( log, comm, shift_index, wrap_procs_devices );
    }
};

}
}

#endif
