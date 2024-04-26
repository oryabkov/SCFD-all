#ifndef __SCFD_MPI_COMMUNICATOR_INFO_H__
#define __SCFD_MPI_COMMUNICATOR_INFO_H__

#include <mpi.h>

namespace scfd
{
namespace communication
{

namespace detail
{

template<class T>
void all_reduce(T *loc_data,T *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    throw std::logic_error("all_reduce:type not implemented");
}

template<>
void all_reduce<float>(float *loc_data,float *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_FLOAT, op, communicator);
}

template<>
void all_reduce<double>(double *loc_data,double *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_DOUBLE, op, communicator);
}

template<>
void all_reduce<int>(int *loc_data,int *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_INT, op, communicator);
}

template<>
void all_reduce<unsigned int>(unsigned int *loc_data,unsigned int *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED, op, communicator);
}

template<>
void all_reduce<long>(long *loc_data,long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_LONG, op, communicator);
}

template<>
void all_reduce<unsigned long>(unsigned long *loc_data,unsigned long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED_LONG, op, communicator);
}

template<>
void all_reduce<long long>(long long *loc_data,long long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_LONG_LONG, op, communicator);
}

template<>
void all_reduce<unsigned long long>(unsigned long long *loc_data,unsigned long long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED_LONG_LONG, op, communicator);
}

} // namespace detail

template<class Ord>
struct mpi_communicator_info
{

    MPI_Comm comm;
    Ord num_procs;
    Ord myid;

    void barrier()const
    {
        MPI_Barrier(comm);
    }

    template<class T>
    void all_reduce(T *loc_data,T *res_data,int count,MPI_Op op)const
    {
        detail::all_reduce(loc_data,res_data,count,op,comm);
    }
    template<class T>
    void all_reduce_sum(T *loc_data,T *res_data,int count)const
    {
        all_reduce(loc_data,res_data,count,MPI_SUM);
    }
    template<class T>
    void all_reduce_max(T *loc_data,T *res_data,int count)const
    {
        all_reduce(loc_data,res_data,count,MPI_MAX);
    }
    template<class T>
    void all_reduce_min(T *loc_data,T *res_data,int count)const
    {
        all_reduce(loc_data,res_data,count,MPI_MIN);
    }
    /// Shortcuts for 1 value
    template<class T>
    T all_reduce_sum(T loc_val)const
    {
        T res_val;
        all_reduce_sum(&loc_val,&res_val,1);
        return res_val;
    }
    template<class T>
    T all_reduce_max(T loc_val)const
    {
        T res_val;
        all_reduce_max(&loc_val,&res_val,1);
        return res_val;
    }
    template<class T>
    T all_reduce_min(T loc_val)const
    {
        T res_val;
        all_reduce_min(&loc_val,&res_val,1);
        return res_val;
    }
};

} // namespace communication
} // namespace scfd


#endif