#ifndef __SCFD_MPI_COMMUNICATOR_INFO_H__
#define __SCFD_MPI_COMMUNICATOR_INFO_H__

#include <mpi.h>

namespace scfd
{
namespace communication
{

template<class Ord>
struct mpi_communicator_info
{

    MPI_Comm comm;
    Ord num_procs;
    Ord myid;

};

} // namespace communication
} // namespace scfd


#endif