#ifndef __SCFD_MPI_COMM_INFO_H__
#define __SCFD_MPI_COMM_INFO_H__

#include <stdexcept>
#include <mpi.h>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define SCFD_MPI_SAFE_CALL(X) \
    do {                                                                                                                                                                                        \
        auto mpi_res = (X);                                                                                                                                                       \
        if (mpi_res != MPI_SUCCESS) throw scfd::communication::mpi_error(mpi_res, std::string("MPI_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: "));    \
    } while (0)

/*#define SCFD_MPI_SAFE_CALL(X,MSG_PREFIX) \
    do {                                                                                                                                                                                        \
        auto mpi_res = (X);                                                                                                                                                       \
        if (mpi_res != MPI_SUCCESS) throw scfd::communication::mpi_error(mpi_res, std::string(MSG_PREFIX "MPI_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: "));    \
    } while (0)

#define SCFD_MPI_SAFE_CALL(X) SCFD_MPI_SAFE_CALL(X,"")*/

namespace scfd
{
namespace communication
{

std::string mpi_error_code_to_str(int error)
{
    if (error == MPI_SUCCESS)
        return std::string("success(") + std::to_string(error) + std::string(")");
    else if (error == MPI_ERR_COMM)
        return std::string("Invalid communicator. A common error is to use a null communicator in a call (not even allowed in MPI_Comm_rank). (") + std::to_string(error) + std::string(")");
    else if (error == MPI_ERR_BUFFER)
        return std::string("Invalid buffer pointer. Usually a null buffer where one is not valid. (") + std::to_string(error) + std::string(")");
    else if (error == MPI_ERR_COUNT)
        return std::string("Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. (") + std::to_string(error) + std::string(")");
    else if (error == MPI_ERR_TYPE)
        return std::string("Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). (") + std::to_string(error) + std::string(")");
    else if (error == MPI_ERR_OP)
        return std::string("Invalid operation. MPI operations (objects of type MPI_Op) must either be one of the predefined operations (e.g., MPI_SUM) or created with MPI_Op_create. (") + std::to_string(error) + std::string(")");
    else 
        return std::string("unknown error(") + std::to_string(error) + std::string(")");
}

struct mpi_error : public std::runtime_error
{
    mpi_error() : std::runtime_error("mpi success"), error_code_(MPI_SUCCESS)
    {
    }
    mpi_error(int error_code, const std::string &msg_prefix = "") : 
      std::runtime_error(msg_prefix + mpi_error_code_to_str(error_code)),
      error_code_(error_code)
    {
    }

    int error_code()const { return error_code_; }
protected:
    int error_code_;
};

namespace detail
{

template<class T>
struct mpi_data_type
{
    static_assert(!std::is_same<T,T>::value, "mpi_data_type:type not implemented");
};

template<>
struct mpi_data_type<char>
{
    static MPI_Datatype mpi_type() { return MPI_BYTE; }
};

template<>
struct mpi_data_type<int>
{
    static MPI_Datatype mpi_type() { return MPI_INT; }
};

template<>
struct mpi_data_type<unsigned int>
{
    static MPI_Datatype mpi_type() { return MPI_UNSIGNED; }
};

template<>
struct mpi_data_type<long>
{
    static MPI_Datatype mpi_type() { return MPI_LONG; }
};

template<>
struct mpi_data_type<unsigned long>
{
    static MPI_Datatype mpi_type() { return MPI_UNSIGNED_LONG; }
};

template<>
struct mpi_data_type<long long>
{
    static MPI_Datatype mpi_type() { return MPI_LONG_LONG; }
};

template<>
struct mpi_data_type<unsigned long long>
{
    static MPI_Datatype mpi_type() { return MPI_UNSIGNED_LONG_LONG; }
};

template<>
struct mpi_data_type<float>
{
    static MPI_Datatype mpi_type() { return MPI_FLOAT; }
};

template<>
struct mpi_data_type<double>
{
    static MPI_Datatype mpi_type() { return MPI_DOUBLE; }
};

template<class T>
void all_gather(
    const T *sendbuf, int sendcount, T *recvbuf, 
    int recvcount, MPI_Comm comm
)
{
    SCFD_MPI_SAFE_CALL( 
        MPI_Allgather(
            static_cast<const void*>(sendbuf), sendcount*sizeof(T), MPI_BYTE,
            static_cast<void*>(recvbuf), recvcount*sizeof(T), MPI_BYTE, comm
        )
    );
}


template<class T>
void all_gatherv(
    const T *sendbuf, int sendcount, T *recvbuf, 
    const int *recvcounts, const int *displs, MPI_Comm comm
)
{
    SCFD_MPI_SAFE_CALL( 
        MPI_Allgatherv(
            static_cast<const void*>(sendbuf), sendcount, mpi_data_type<T>::mpi_type(),
            static_cast<void*>(recvbuf), recvcounts, displs, mpi_data_type<T>::mpi_type(), comm
        )
    );
}


template<class T>
void all_reduce(T *loc_data,T *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    throw std::logic_error("all_reduce:type not implemented");
}

template<>
void all_reduce<float>(float *loc_data,float *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_FLOAT, op, communicator) );
}

template<>
void all_reduce<double>(double *loc_data,double *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_DOUBLE, op, communicator) );
}

template<>
void all_reduce<int>(int *loc_data,int *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_INT, op, communicator) );
}

template<>
void all_reduce<unsigned int>(unsigned int *loc_data,unsigned int *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED, op, communicator) );
}

template<>
void all_reduce<long>(long *loc_data,long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_LONG, op, communicator) );
}

template<>
void all_reduce<unsigned long>(unsigned long *loc_data,unsigned long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED_LONG, op, communicator) );
}

template<>
void all_reduce<long long>(long long *loc_data,long long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_LONG_LONG, op, communicator) );
}

template<>
void all_reduce<unsigned long long>(unsigned long long *loc_data,unsigned long long *res_data,int count,MPI_Op op,MPI_Comm communicator)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(static_cast<void*>(loc_data), static_cast<void*>(res_data), count, MPI_UNSIGNED_LONG_LONG, op, communicator) );
}

} // namespace detail

class mpi_comm;

struct mpi_comm_info
{

    MPI_Comm comm;
    int num_procs;
    int myid;

    void barrier()const
    {
        SCFD_MPI_SAFE_CALL( MPI_Barrier(comm) );
    }

    template<class T>
    void all_gather(
        const T *sendbuf, int sendcount, T *recvbuf, int recvcount
    )const
    {
        detail::all_gather(sendbuf, sendcount, recvbuf, recvcount, comm);
    }
    template<class T>
    void all_gatherv(
        const T *sendbuf, int sendcount, T *recvbuf, 
        const int *recvcounts, const int *displs
    )const
    {
        detail::all_gatherv(sendbuf, sendcount, recvbuf, recvcounts, displs, comm);
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

    mpi_comm split(int color, int key)const;
    mpi_comm split(int color)const;
    mpi_comm split_type(int color, int key)const;
    mpi_comm split_type(int color)const;

};

} // namespace communication
} // namespace scfd


#endif