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

#ifndef __SCFD_PLATFORM_SYCL_MPI_H__
#define __SCFD_PLATFORM_SYCL_MPI_H__

#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <scfd/backend/sycl.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

namespace scfd
{
namespace platform
{

template <class Ordinal = PLATFORM_ORDINAL, class BigOrdinal = PLATFORM_BIG_ORDINAL>
struct sycl_mpi
{
    using backend_type                   = backend::sycl<Ordinal>;
    using ordinal_type                   = Ordinal;
    using big_ordinal_type               = BigOrdinal;
    using communicator_type              = scfd::communication::mpi_comm_info;
    using communication_environment_type = scfd::communication::mpi_wrap;

    template <class Log>
    static int init( Log &log, const communicator_type &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        auto node_comm = comm.split_type( MPI_COMM_TYPE_SHARED );
        int  node_size = node_comm.num_procs();
        int  my_id     = node_comm.myid();
        node_comm.free();

        std::vector<::sycl::device> devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
        const int                   number_of_devices_on_node = static_cast<int>( devices.size() );
        if ( number_of_devices_on_node <= 0 )
            throw std::runtime_error( "sycl_mpi::init: no visible SYCL GPU devices" );
        if ( number_of_devices_on_node < node_size && !wrap_procs_devices )
        {
            throw std::runtime_error(
                "sycl_mpi::init: number of nproc = " + std::to_string( node_size ) + ", number of SYCL GPU devices = " +
                std::to_string( number_of_devices_on_node ) + "\n numproc per node > numDevices per node"
            );
        }

        const int device_id = ( my_id + shift_index ) % number_of_devices_on_node;
        if ( number_of_devices_on_node < node_size && wrap_procs_devices && my_id == 0 )
        {
            log.warning_f(
                "sycl_mpi::init is wrapping %i MPI processes over %i visible device(s). "
                "Several MPI processes will share one device.",
                node_size, number_of_devices_on_node
            );
        }
        log.info_all_f(
            "sycl_mpi::init: global_size = %i, global_id = %i, node_size = %i, devices_on_node = %i, "
            "node_device_id = %i, node_my_id = %i",
            comm.num_procs, comm.myid, node_size, number_of_devices_on_node, device_id, my_id
        );
        sycl_device_queue = ::sycl::queue( devices[static_cast<std::size_t>( device_id )] );
        return device_id;
    }

    static int init( const communicator_type &comm, int shift_index = 0, bool wrap_procs_devices = false )
    {
        scfd::utils::log_mpi log( comm.comm );
        return init( log, comm, shift_index, wrap_procs_devices );
    }
};

}
}

#endif
