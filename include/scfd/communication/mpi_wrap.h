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

#ifndef __SCFD_MPI_WRAP_H__
#define __SCFD_MPI_WRAP_H__

#include <mpi.h>
#include "mpi_comm_info.h"

namespace scfd
{
namespace communication
{

template<std::int8_t UseThreadInit = -1>
struct mpi_wrap
{
    // https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Init_thread.3.html
    // UseThreadInit=0: MPI_THREAD_SINGLE: Indicating that only one thread will execute.
    // UseThreadInit=1: MPI_THREAD_FUNNELED: Indicating that if the process is multithreaded, only the thread that called MPI_Init_thread will make MPI calls.
    // UseThreadInit=2: MPI_THREAD_SERIALIZED: Indicating that if the process is multithreaded, only one thread will make MPI library calls at one time.
    // UseThreadInit=3: MPI_THREAD_MULTIPLE: Indicating that if the process is multithreaded, multiple threads may call MPI at once with no restrictions.
    
    mpi_wrap(int argc, char *argv[]):
    provided_threads_(-1)
    {
        if constexpr(UseThreadInit == 0)
        {
            MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_threads_);
        }
        else if constexpr(UseThreadInit == 1)
        {
            MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_threads_);
        }
        else if constexpr(UseThreadInit == 2)
        {
            MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided_threads_);
        }
        else if constexpr(UseThreadInit == 3)
        {
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_threads_);
        }        
        else
        {
            MPI_Init(&argc, &argv);
        }
        //int provided;
        //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        /*if (provided != MPI_THREAD_FUNNELED)
        {
            std::cout << "WARNING: mpi_wrap: provided != MPI_THREAD_FUNNELED" << std::endl;
        }*/
        data.comm = MPI_COMM_WORLD;
        MPI_Comm_size(MPI_COMM_WORLD, &data.num_procs );
        MPI_Comm_rank(MPI_COMM_WORLD, &data.myid );
        data.provided_threads = provided_threads_;
    }
    ~mpi_wrap()
    {
        MPI_Finalize();
    }

    ///NOTE please use this method instead of data public member
    mpi_comm_info comm_world()const { return data; }    
    int provided_threads()const {return provided_threads_; }

    mpi_comm_info data;  

private:
    int provided_threads_; 

};

} // namespace communication
} // namespace scfd


#endif