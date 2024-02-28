#ifndef __SCFD_MPI_RECT_DISTRIBUTOR_H__
#define __SCFD_MPI_RECT_DISTRIBUTOR_H__

#include <vector>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include <scfd/arrays/array_nd.h>
#include "mpi_communicator_info.h"
#include "rect_partitioner.h"

namespace scfd
{
namespace communication
{

template<class Ord,class BigOrd,int Dim>
struct mpi_rect_distributor
{
    typedef     static_vec::vec<BigOrd,Dim>         big_ord_vec_t;
    typedef     static_vec::rect<BigOrd,Dim>        big_ord_rect_t;
    typedef     rect_partitioner<Ord,BigOrd,Dim>    rect_partitioner_t;

    mpi_communicator_info<Ord>      comm_info;

    mpi_rect_distributor() = default;
    mpi_rect_distributor(const rect_partitioner_t &partitioner)
    {
        init(partitioner);
    }
    void init(const rect_partitioner_t &partitioner)
    {
        comm_info = partitioner.comm_info;
        //TODO 
    }
    
    Ord     get_own_rank()const { return comm_info.myid; }

    template<class T, class Memory>
    void    sync(const arrays::array_nd<T,dim,Memory> &array)
    {
        //TODO
    }
};

} // namespace communication
} // namespace scfd

#endif
