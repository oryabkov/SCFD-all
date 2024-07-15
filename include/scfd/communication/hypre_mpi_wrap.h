#ifndef __SCFD_HYPRE_MPI_WRAP_H__
#define __SCFD_HYPRE_MPI_WRAP_H__

#include "_hypre_utilities.h"
#include "mpi_comm_info.h"

namespace scfd
{
namespace communication
{

struct hypre_mpi_wrap
{


    hypre_mpi_wrap(int argc, char *argv[])
    {
        hypre_MPI_Init(&argc, &argv);
        //int provided;
        //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        /*if (provided != MPI_THREAD_FUNNELED)
        {
            std::cout << "WARNING: hypre_mpi_wrap: provided != MPI_THREAD_FUNNELED" << std::endl;
        }*/
        data.comm = hypre_MPI_COMM_WORLD;
        hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &data.num_procs );
        hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &data.myid );   
    }
    ~hypre_mpi_wrap()
    {
        hypre_MPI_Finalize();
        //MPI_Finalize();
    }

    ///NOTE please use this method instead of data public member
    mpi_comm_info comm_world()const { return data; }    

    mpi_comm_info data;
    

};

} // namespace communication
} // namespace scfd

#endif