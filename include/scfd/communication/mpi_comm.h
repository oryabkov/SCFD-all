#ifndef __SCFD_MPI_COMM_H__
#define __SCFD_MPI_COMM_H__

#include <stdexcept>
#include <mpi.h>
#include "mpi_comm_info.h"




namespace scfd
{
namespace communication
{

struct mpi_comm
{
    mpi_comm(MPI_Comm comm) : comm_(comm)
    {
        if (comm_ == MPI_COMM_NULL) 
        {
            /// Just for reference
            num_procs_ = 0;
            myid_ = 0;
        }
        else
        {
            SCFD_MPI_SAFE_CALL( MPI_Comm_size(comm_, &num_procs_ ) );
            SCFD_MPI_SAFE_CALL( MPI_Comm_rank(comm_, &myid_ ) );
        }
    }
    mpi_comm(const mpi_comm &) = delete;
    mpi_comm(mpi_comm &&c)
    {
        num_procs_ = c.num_procs_;
        myid_ = c.myid_;
        comm_ = c.comm_;
        c.comm_ = MPI_COMM_NULL;
    }
    const mpi_comm &operator=(const mpi_comm &) = delete;
    const mpi_comm &operator=(mpi_comm &&c)
    {
        free();
        num_procs_ = c.num_procs_;
        myid_ = c.myid_;
        comm_ = c.comm_;
        c.comm_ = MPI_COMM_NULL;
        return *this;
    }
    ~mpi_comm()
    {
        free();
    }

    void free()
    {
        if (comm_ == MPI_COMM_NULL) return;
        SCFD_MPI_SAFE_CALL( MPI_Comm_free(&comm_) );
    }

    MPI_Comm comm()const { return comm_; }
    int num_procs()const { return num_procs_; }
    int myid()const { return myid_; }
    mpi_comm_info info()const
    {
        ///ISSUE leave info as plain struct or make it to be class??
        return mpi_comm_info{comm_,num_procs_,myid_};
    }

private:
    MPI_Comm comm_;
    int num_procs_;
    int myid_;
};

mpi_comm mpi_comm_info::split(int color, int key)const
{
    MPI_Comm newcomm;
    SCFD_MPI_SAFE_CALL( MPI_Comm_split(comm, color, key, &newcomm) );
    return mpi_comm(newcomm);
}

mpi_comm mpi_comm_info::split(int color)const
{
    return split(color, myid);
}

} // namespace communication
} // namespace scfd


#endif