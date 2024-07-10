#ifndef __SCFD_MPI_WRAP_H__
#define __SCFD_MPI_WRAP_H__

#include <mpi.h>
#include "mpi_comm_info.h"

namespace scfd
{
namespace communication
{

struct mpi_wrap
{


    mpi_wrap(int argc, char *argv[])
    {
        MPI_Init(&argc, &argv);
        //int provided;
        //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        /*if (provided != MPI_THREAD_FUNNELED)
        {
            std::cout << "WARNING: mpi_wrap: provided != MPI_THREAD_FUNNELED" << std::endl;
        }*/
        data.comm = MPI_COMM_WORLD;
        MPI_Comm_size(MPI_COMM_WORLD, &data.num_procs );
        MPI_Comm_rank(MPI_COMM_WORLD, &data.myid );   
    }
    ~mpi_wrap()
    {
        MPI_Finalize();
    }

    ///NOTE please use this method instead of data public member
    mpi_comm_info comm_world()const { return data; }    

    mpi_comm_info data;    

};

} // namespace communication
} // namespace scfd


#endif