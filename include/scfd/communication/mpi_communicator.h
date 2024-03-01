#ifndef __SCFD_MPI_COMMUNICATOR_H__
#define __SCFD_MPI_COMMUNICATOR_H__

#include <mpi.h>
#include "mpi_communicator_info.h"

namespace scfd
{
namespace communication
{

template<class Ord>
struct mpi_communicator
{


    mpi_communicator(int argc, char *argv[])
    {
        MPI_Init(&argc, &argv);
        //int provided;
        //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        /*if (provided != MPI_THREAD_FUNNELED)
        {
            std::cout << "WARNING: mpi_communicator: provided != MPI_THREAD_FUNNELED" << std::endl;
        }*/
        data.comm = MPI_COMM_WORLD;
        MPI_Comm_size(MPI_COMM_WORLD, &data.num_procs );
        MPI_Comm_rank(MPI_COMM_WORLD, &data.myid );   
    }
    ~mpi_communicator()
    {
        MPI_Finalize();
    }

    mpi_communicator_info<Ord> data;    

};

} // namespace communication
} // namespace scfd


#endif