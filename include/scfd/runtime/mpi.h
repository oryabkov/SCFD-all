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

#ifndef __SCFD_BACKEND_RUNTIME_MPI_H__
#define __SCFD_BACKEND_RUNTIME_MPI_H__

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>
#include <scfd/backend/backend.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/log_std.h>

#if defined( PLATFORM_CUDA )
#    include <scfd/utils/init_cuda_mpi.h>
#elif defined( PLATFORM_HIP )
#    include <scfd/utils/init_hip_mpi.h>
#endif

namespace scfd
{
namespace backend
{
namespace detail
{

#if defined( PLATFORM_CUDA )
template <bool WrapProcsDevices, class Log, class Comm>
int cuda_runtime::init_device_mpi( Log &log, const Comm &comm, int shift_index )
{
    return scfd::utils::init_cuda_mpi<Log, WrapProcsDevices>( log, comm, shift_index );
}

template <bool WrapProcsDevices, class Comm>
int cuda_runtime::init_device_mpi( const Comm &comm, int shift_index )
{
    scfd::utils::log_std log;
    return init_device_mpi<WrapProcsDevices>( log, comm, shift_index );
}
#endif

#if defined( PLATFORM_HIP )
template <bool WrapProcsDevices, class Log, class Comm>
int hip_runtime::init_device_mpi( Log &log, const Comm &comm, int shift_index )
{
    return scfd::utils::init_hip_mpi<Log, WrapProcsDevices>( log, comm, shift_index );
}

template <bool WrapProcsDevices, class Comm>
int hip_runtime::init_device_mpi( const Comm &comm, int shift_index )
{
    scfd::utils::log_std log;
    return init_device_mpi<WrapProcsDevices>( log, comm, shift_index );
}
#endif

#if defined( PLATFORM_SYCL )
template <bool WrapProcsDevices, class Log, class Comm>
int sycl_runtime::init_device_mpi( Log &log, const Comm &comm, int shift_index )
{
    auto node_comm = comm.split_type( MPI_COMM_TYPE_SHARED );
    int  node_size = node_comm.num_procs();
    int  my_id     = node_comm.myid();
    node_comm.free();

    std::vector<::sycl::device> devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
    const int                   number_of_devices_on_node = static_cast<int>( devices.size() );
    if ( number_of_devices_on_node <= 0 )
        throw std::runtime_error( "sycl_runtime::init_device_mpi: no visible SYCL GPU devices" );
    if ( number_of_devices_on_node < node_size && !WrapProcsDevices )
    {
        throw std::runtime_error(
            "sycl_runtime::init_device_mpi: number of nproc = " + std::to_string( node_size ) +
            ", number of SYCL GPU devices = " + std::to_string( number_of_devices_on_node ) +
            "\n numproc per node > numDevices per node"
        );
    }

    const int device_id = ( my_id + shift_index ) % number_of_devices_on_node;
    if ( number_of_devices_on_node < node_size && WrapProcsDevices && my_id == 0 )
    {
        log.info_f(
            "WARNING: sycl_runtime::init_device_mpi is wrapping %i MPI processes over %i visible device(s). "
            "Several MPI processes will share one device.",
            node_size, number_of_devices_on_node
        );
    }
    log.info_f(
        "sycl_runtime::init_device_mpi: global_size = %i, global_id = %i, node_size = %i, devices_on_node = %i, "
        "node_device_id = %i, node_my_id = %i",
        comm.num_procs, comm.myid, node_size, number_of_devices_on_node, device_id, my_id
    );
    sycl_device_queue = ::sycl::queue( devices[static_cast<std::size_t>( device_id )] );
    return device_id;
}

template <bool WrapProcsDevices, class Comm>
int sycl_runtime::init_device_mpi( const Comm &comm, int shift_index )
{
    scfd::utils::log_std log;
    return init_device_mpi<WrapProcsDevices>( log, comm, shift_index );
}
#endif

}
}
}

#endif
