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

#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/sequential_mpi_debug.h>
#include <iostream>



int main(int argc, char *argv[])
{
    scfd::communication::mpi_wrap mpi_wrap(argc, argv);
    auto mpi_comm = mpi_wrap.comm_world();
    scfd::communication::sequential_mpi_debug seq_mpi(mpi_comm);

    //this marks the section which will be executed in sequential order, from 0 to mpi_comm.num_procs
    if(mpi_comm.myid == 0)
    {
        std::cout << "the following output should be in sequential order of proc = ..., and proc_running = ..." << std::endl;
    }
    seq_mpi.start(); 
    auto proc_running = seq_mpi.get_rank();
    std::cout << "proc = " << mpi_comm.myid << ", proc_running = " << proc_running << std::endl;
    seq_mpi.stop();
    std::cout << "after_barrier: " << "proc = " << mpi_comm.myid << ", proc_running = " << proc_running << std::endl;


    return 0;
}