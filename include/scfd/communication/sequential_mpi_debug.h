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

#ifndef __SCFD_SEQUENTIAL_MPI_DEBUG_H__
#define __SCFD_SEQUENTIAL_MPI_DEBUG_H__

#include <mpi.h>
#include "mpi_comm_info.h"


/**
 * @brief      This is a debuging class which forced the processes to be executed in sequential order even on distributed machines
 *             Tested on multiple GPU msu270 cluster.
 *
 * @tparam     MPI   { is the SCFD mpi_comm class, but can be its substitusion that requires the same signature as the aforementioned. }
 */
namespace scfd
{
namespace communication
{

template<class MPI>
class sequential_mpi_debug
{
public:
    sequential_mpi_debug(const MPI& mpi):
    mpi_comm_(mpi),
    tag(1),
    proc_running(0)
    {}
    ~sequential_mpi_debug()
    {}

    //there should be no barriers between start and stop!
    void start() 
    {
        mpi_comm_.barrier(); 
        if(mpi_comm_.myid != 0)
        {
            MPI_Status status;
            MPI_Request request;
            int source = mpi_comm_.myid - 1; //cascading recieve from

            SCFD_MPI_SAFE_CALL(
                MPI_Irecv(&proc_running, 1, MPI_INT,
                source, tag, mpi_comm_.comm, &request)
            );
            SCFD_MPI_SAFE_CALL(
                MPI_Wait(&request, &status)
            );

        }
    }
    //there should be no barriers between start and stop!    
    void stop()
    {
        int destination = mpi_comm_.myid + 1; //cascading sends
        proc_running++;
        if(mpi_comm_.myid < mpi_comm_.num_procs - 1)
        {
            SCFD_MPI_SAFE_CALL(
                MPI_Send(&proc_running, 1, MPI_INT, destination,
                tag, mpi_comm_.comm)
            );
        }
        mpi_comm_.barrier();        
    }

    int get_rank()
    {
        return proc_running;
    }


    
private:
    const MPI& mpi_comm_;
    int tag;
    int proc_running;

};

}
}

#endif