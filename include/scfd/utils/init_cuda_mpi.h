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

template<class Log>
inline int init_cuda_mpi(Log &log, const scfd::communication::mpi_comm_info& comm, int shift_index = 0)
{
    int node_size = 0;
    int my_id = 0;
    int number_of_devices_on_node = 0;
    int device_id = 0;

    char node_name[255];
    int node_name_size = 0;
    SCFD_MPI_SAFE_CALL( MPI_Get_processor_name( node_name, &node_name_size ) );
    int char_hash = 0;
    for(int j=0;j<node_name_size;j++)
    {
        char_hash = char_hash + static_cast<int>(node_name[j]);
    }    
    auto comm_split = comm.split(char_hash);
    node_size = comm_split.num_procs();
    my_id = comm_split.myid(); 
    comm_split.free();   
    CUDA_SAFE_CALL( cudaGetDeviceCount(&number_of_devices_on_node) );
    if(number_of_devices_on_node<node_size)
    {
        throw std::runtime_error("init_cuda_mpi: node name " + std::string(node_name) + "\n number of nproc = " + std::to_string(node_size) + ", number of GPUs = " + std::to_string(number_of_devices_on_node) + "\n numproc per node > numGPUs per node!" );
    }
    device_id = (my_id + shift_index) % number_of_devices_on_node;
    log.info_f("init_cuda_mpi split_color: node_name = %s, node_color = %i, global_size = %i, global_id = %i, node_size = %i, devices_on_node = %i, node_device_id = %i, node_my_id = %i", node_name, char_hash, comm.num_procs, comm.myid, node_size, number_of_devices_on_node, device_id, my_id); 
    return scfd::utils::init_cuda(-2, device_id);
}


inline int init_cuda_mpi(const scfd::communication::mpi_comm_info& comm, int shift_index = 0)
{
    log_std log;
    return init_cuda_mpi(log, comm, shift_index);
}


}
}

#endif