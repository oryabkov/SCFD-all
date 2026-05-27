// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_INITCUDAMPI_H__
#define __SCFD_UTILS_INITCUDAMPI_H__

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/utils/log_std.h>
#include <scfd/utils/init_cuda.h>

namespace scfd
{
namespace utils
{

inline bool spsfd_device_bind_debug_enabled()
{
    const char *env = std::getenv( "SPSFD_DEVICE_BIND_DEBUG" );
    return env && env[0] && env[0] != '0';
}

inline const char *spsfd_env_or_na( const char *name )
{
    const char *value = std::getenv( name );
    return value && value[0] ? value : "n/a";
}

template <class Log, bool WrapProcsGPUs = false>
inline int init_cuda_mpi( Log &log, const scfd::communication::mpi_comm_info &comm, int shift_index = 0 )
{
    int node_size                 = 0;
    int my_id                     = 0;
    int number_of_devices_on_node = 0;
    int device_id                 = 0;

    char node_name[255];
    int  node_name_size = 0;
    SCFD_MPI_SAFE_CALL( MPI_Get_processor_name( node_name, &node_name_size ) );
    auto comm_split = comm.split_type( MPI_COMM_TYPE_SHARED );
    node_size       = comm_split.num_procs();
    my_id           = comm_split.myid();
    comm_split.free();
    CUDA_SAFE_CALL( cudaGetDeviceCount( &number_of_devices_on_node ) );
    if ( number_of_devices_on_node <= 0 )
    {
        throw std::runtime_error(
            "init_cuda_mpi: node name " + std::string( node_name ) + "\n no visible CUDA devices"
        );
    }
    if ( number_of_devices_on_node < node_size && !WrapProcsGPUs )
    {
        throw std::runtime_error(
            "init_cuda_mpi: node name " + std::string( node_name ) + "\n number of nproc = " +
            std::to_string( node_size ) + ", number of GPUs = " + std::to_string( number_of_devices_on_node ) +
            "\n numproc per node > numGPUs per node!"
        );
    }
    device_id = ( my_id + shift_index ) % number_of_devices_on_node;
    if ( number_of_devices_on_node < node_size && WrapProcsGPUs && my_id == 0 )
    {
        log.info_f(
            "WARNING: init_cuda_mpi is wrapping %i MPI processes over %i visible GPU(s) on node %s. "
            "Several MPI processes will share one GPU.",
            node_size, number_of_devices_on_node, node_name
        );
    }
    log.info_f(
        "init_cuda_mpi split_type_shared: node_name = %s, global_size = %i, global_id = %i, node_size = %i, "
        "devices_on_node = %i, node_device_id = %i, node_my_id = %i",
        node_name, comm.num_procs, comm.myid, node_size, number_of_devices_on_node, device_id, my_id
    );
    const int initialized_device = scfd::utils::init_cuda( -2, device_id );

    if ( spsfd_device_bind_debug_enabled() )
    {
        int         active_device = -1;
        char        pci_bus_id[64] = "n/a";
        const char *device_name = "n/a";
        std::size_t free_mem = 0;
        std::size_t total_mem = 0;

        CUDA_SAFE_CALL( cudaGetDevice( &active_device ) );
        if ( active_device >= 0 )
        {
            cudaDeviceProp prop;
            CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, active_device ) );
            device_name = prop.name;
            CUDA_SAFE_CALL( cudaDeviceGetPCIBusId( pci_bus_id, sizeof( pci_bus_id ), active_device ) );
            CUDA_SAFE_CALL( cudaMemGetInfo( &free_mem, &total_mem ) );
        }

        std::fprintf(
            stderr,
            "[SPSFD_DEVICE_BIND_DEBUG SCFD_CUDA_BIND] rank=%i/%i local_rank=%i/%i "
            "visible_devices=%i requested_device=%i active_device=%i initialized_device=%i "
            "pci=%s name=\"%s\" free_MB=%.3f total_MB=%.3f host=%s "
            "CUDA_VISIBLE_DEVICES=%s NVIDIA_VISIBLE_DEVICES=%s SLURM_PROCID=%s "
            "SLURM_LOCALID=%s SLURM_NODEID=%s PMI_RANK=%s PMIX_RANK=%s\n",
            comm.myid, comm.num_procs, my_id, node_size, number_of_devices_on_node, device_id,
            active_device, initialized_device, pci_bus_id, device_name,
            static_cast<double>( free_mem ) / ( 1024.0 * 1024.0 ),
            static_cast<double>( total_mem ) / ( 1024.0 * 1024.0 ), node_name,
            spsfd_env_or_na( "CUDA_VISIBLE_DEVICES" ), spsfd_env_or_na( "NVIDIA_VISIBLE_DEVICES" ),
            spsfd_env_or_na( "SLURM_PROCID" ), spsfd_env_or_na( "SLURM_LOCALID" ),
            spsfd_env_or_na( "SLURM_NODEID" ), spsfd_env_or_na( "PMI_RANK" ),
            spsfd_env_or_na( "PMIX_RANK" )
        );
        std::fflush( stderr );
    }

    return initialized_device;
}


template <bool WrapProcsGPUs = false>
inline int init_cuda_mpi( const scfd::communication::mpi_comm_info &comm, int shift_index = 0 )
{
    log_std log;
    return init_cuda_mpi<log_std, WrapProcsGPUs>( log, comm, shift_index );
}


}
}

#endif
