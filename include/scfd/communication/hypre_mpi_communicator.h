#ifndef __SCFD_HYPRE_MPI_COMMUNICATOR_H__
#define __SCFD_HYPRE_MPI_COMMUNICATOR_H__

#include "_hypre_utilities.h"
#include <mpi_communicator_info.h>

namespace scfd
{
namespace communication
{

template<class Ord>
struct hypre_mpi_communicator
{


    hypre_mpi_communicator(int argc, char *argv[])
    {
        hypre_MPI_Init(&argc, &argv);
        //int provided;
        //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        /*if (provided != MPI_THREAD_FUNNELED)
        {
            std::cout << "WARNING: hypre_mpi_communicator: provided != MPI_THREAD_FUNNELED" << std::endl;
        }*/
        data.comm = hypre_MPI_COMM_WORLD;
        hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &data.num_procs );
        hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &data.myid );   
    }
    ~hypre_mpi_communicator()
    {
        hypre_MPI_Finalize();
        //MPI_Finalize();
    }

    mpi_communicator_info<Ord> data;
    

};

} // namespace communication
} // namespace scfd

#endif