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
#include <scfd/communication/mpi_comm_info.h>

using comm_info_type = scfd::communication::mpi_comm_info;

int main(int argc, char *argv[]) {

    int thread = -1;
    if(argc == 2)
    {
        thread = std::stoi(argv[1]);
    }
    scfd::communication::mpi_wrap mpi(argc, argv, thread);
    comm_info_type comm_info = mpi.comm_world();

    int provided_threads = mpi.provided_threads();
    if(thread != -1)
    {
        std::cout << "provided_threads: " << provided_threads << std::endl;
    }

    if((comm_info.myid == 0)&&(thread == -1))
    {
        std::cout << "Usage: " << argv[0] << " thread" << std::endl;
        std::cout << " possible thread values:" << std::endl;
        std::cout << " thread=-1 (default): running MPI_Init(argc,argv)." << std::endl;        
        std::cout << " thread=0: MPI_THREAD_SINGLE: Indicating that only one thread will execute." << std::endl;
        std::cout << " thread=1: MPI_THREAD_FUNNELED: Indicating that if the process is multithreaded, only the thread that called MPI_Init_thread will make MPI calls." << std::endl;
        std::cout << " thread=2: MPI_THREAD_SERIALIZED: Indicating that if the process is multithreaded, only one thread will make MPI library calls at one time." << std::endl;
        std::cout << " thread=3: MPI_THREAD_MULTIPLE: Indicating that if the process is multithreaded, multiple threads may call MPI at once with no restrictions." << std::endl;        
    }

    return 0;
}