// Copyright © 2023-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#include <iostream>
#include <scfd/utils/log_mpi.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm.h>

int main(int argc, char *argv[]) 
{
    using mpi_comm_t = scfd::communication::mpi_comm;
    using mpi_comm_info_t = typename mpi_comm_t::mpi_comm_info_type;
    using log_t = scfd::utils::log_mpi;
    using mpi_wrap_t = scfd::communication::mpi_wrap;

    mpi_wrap_t mpi(argc, argv);
    mpi_comm_info_t comm_info = mpi.comm_world();
    log_t log;

    auto num_procs = comm_info.num_procs;
    auto my_id = comm_info.myid;
    
    int color = my_id/2;
    mpi_comm_t comm1 = comm_info.split(color);
    int numproc1 = comm1.num_procs();
    int my_id1 = comm1.myid();
    auto comm1_info = comm1.info();

    auto res1 = comm1_info.all_reduce_sum(1);
    auto res = comm_info.all_reduce_sum(1);
    auto res_all = comm_info.all_reduce_sum(res1);
    int res_all_ref = 2*num_procs;

    if(num_procs%2 == 1)
    {
        res_all_ref = 2*num_procs - 1;
    }
    
    log.info_f("res_all = %i, res_all_ref = %i", res_all, res_all_ref);
    if(res_all_ref != res_all)
    {
        log.error("FAILED");
        return 1;
    }
    else
    {
        log.info_f("PASSED");
        return 0;    
    }
    
}