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

#ifndef __SCFD_MPI_COMM_INFO_H__
#define __SCFD_MPI_COMM_INFO_H__

#include <stdexcept>
#include <string>
#include <type_traits>
#include <mpi.h>

#define __STR_HELPER( x ) #x
#define __STR( x ) __STR_HELPER( x )

#define SCFD_MPI_SAFE_CALL( X )                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        auto mpi_res = ( X );                                                                                          \
        char err_str[1024];                                                                                            \
        int  str_len;                                                                                                  \
        MPI_Error_string( mpi_res, err_str, &str_len );                                                                \
        if ( mpi_res != MPI_SUCCESS )                                                                                  \
            throw scfd::communication::mpi_error(                                                                      \
                mpi_res, std::string( "MPI_SAFE_CALL " __FILE__ " " __STR( __LINE__ ) " : " #X " failed: " ) +         \
                             std::string( err_str ) + " : "                                                            \
            );                                                                                                         \
    } while ( 0 )

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

std::string mpi_error_code_to_str( int error )
{
    if ( error == MPI_SUCCESS )
        return std::string( "success(" ) + std::to_string( error ) + std::string( ")" );
    else if ( error == MPI_ERR_COMM )
        return std::string( "Invalid communicator. A common error is to use a null communicator in a call (not even "
                            "allowed in MPI_Comm_rank). (" ) +
               std::to_string( error ) + std::string( ")" );
    else if ( error == MPI_ERR_BUFFER )
        return std::string( "Invalid buffer pointer. Usually a null buffer where one is not valid. (" ) +
               std::to_string( error ) + std::string( ")" );
    else if ( error == MPI_ERR_COUNT )
        return std::string(
                   "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. ("
               ) +
               std::to_string( error ) + std::string( ")" );
    else if ( error == MPI_ERR_TYPE )
        return std::string( "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). (" ) +
               std::to_string( error ) + std::string( ")" );
    else if ( error == MPI_ERR_OP )
        return std::string( "Invalid operation. MPI operations (objects of type MPI_Op) must either be one of the "
                            "predefined operations (e.g., MPI_SUM) or created with MPI_Op_create. (" ) +
               std::to_string( error ) + std::string( ")" );
    else
        return std::string( "unknown error(" ) + std::to_string( error ) + std::string( ")" );
}

struct mpi_error : public std::runtime_error
{
    mpi_error() : std::runtime_error( "mpi success" ), error_code_( MPI_SUCCESS )
    {
    }
    mpi_error( int error_code, const std::string &msg_prefix = "" )
        : std::runtime_error( msg_prefix + mpi_error_code_to_str( error_code ) ), error_code_( error_code )
    {
    }

    int error_code() const
    {
        return error_code_;
    }

protected:
    int error_code_;
};

namespace detail
{

struct mpi_request //default zero initialization for simpler constructor operations in mgpu FFT
{
    mpi_request( MPI_Request request_ = MPI_REQUEST_NULL ) : request( request_ )
    {
    }

    MPI_Request *native_ptr()
    {
        return &request;
    }

    const MPI_Request *native_ptr() const
    {
        return &request;
    }

    MPI_Request request;
};

struct mpi_status
{
    mpi_status() : status()
    {
    }

    MPI_Status *native_ptr()
    {
        return &status;
    }

    const MPI_Status *native_ptr() const
    {
        return &status;
    }

    int source() const
    {
        return status.MPI_SOURCE;
    }

    int tag() const
    {
        return status.MPI_TAG;
    }

    int error() const
    {
        return status.MPI_ERROR;
    }

    MPI_Status status;
};

template <class T = void>
struct mpi_data_type
{
    mpi_data_type( MPI_Datatype datatype_ = MPI_DATATYPE_NULL ) : datatype( datatype_ )
    {
    }

    MPI_Datatype native() const
    {
        return datatype;
    }

    MPI_Datatype *native_ptr()
    {
        return &datatype;
    }

    const MPI_Datatype *native_ptr() const
    {
        return &datatype;
    }

    bool is_null() const
    {
        return datatype == MPI_DATATYPE_NULL;
    }

    static mpi_data_type<> mpi_type()
    {
        static_assert( !std::is_same<T, T>::value, "mpi_data_type:type not implemented" );
        return mpi_data_type<>();
    }

    MPI_Datatype datatype;
};

template <>
struct mpi_data_type<char> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_BYTE );
    }
};

template <>
struct mpi_data_type<int> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_INT );
    }
};

template <>
struct mpi_data_type<unsigned int> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_UNSIGNED );
    }
};

template <>
struct mpi_data_type<long> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_LONG );
    }
};

template <>
struct mpi_data_type<unsigned long> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_UNSIGNED_LONG );
    }
};

template <>
struct mpi_data_type<long long> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_LONG_LONG );
    }
};

template <>
struct mpi_data_type<unsigned long long> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_UNSIGNED_LONG_LONG );
    }
};

template <>
struct mpi_data_type<float> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_FLOAT );
    }
};

template <>
struct mpi_data_type<double> : mpi_data_type<>
{
    static mpi_data_type<> mpi_type()
    {
        return mpi_data_type<>( MPI_DOUBLE );
    }
};

static_assert( sizeof( mpi_request ) == sizeof( MPI_Request ), "mpi_request must match MPI_Request size" );
static_assert( alignof( mpi_request ) == alignof( MPI_Request ), "mpi_request must match MPI_Request alignment" );
static_assert( sizeof( mpi_status ) == sizeof( MPI_Status ), "mpi_status must match MPI_Status size" );
static_assert( alignof( mpi_status ) == alignof( MPI_Status ), "mpi_status must match MPI_Status alignment" );
static_assert( sizeof( mpi_data_type<> ) == sizeof( MPI_Datatype ), "mpi_data_type must match MPI_Datatype size" );
static_assert( alignof( mpi_data_type<> ) == alignof( MPI_Datatype ), "mpi_data_type must match MPI_Datatype alignment" );

inline MPI_Request *raw_requests( mpi_request *requests )
{
    return reinterpret_cast<MPI_Request *>( requests );
}

inline const MPI_Request *raw_requests( const mpi_request *requests )
{
    return reinterpret_cast<const MPI_Request *>( requests );
}

inline MPI_Status *raw_statuses( mpi_status *statuses )
{
    return statuses == nullptr ? MPI_STATUSES_IGNORE : reinterpret_cast<MPI_Status *>( statuses );
}

inline const MPI_Status *raw_statuses( const mpi_status *statuses )
{
    return statuses == nullptr ? MPI_STATUSES_IGNORE : reinterpret_cast<const MPI_Status *>( statuses );
}

inline MPI_Status *raw_status( mpi_status *status )
{
    return status == nullptr ? MPI_STATUS_IGNORE : reinterpret_cast<MPI_Status *>( status );
}

inline const MPI_Status *raw_status( const mpi_status *status )
{
    return status == nullptr ? MPI_STATUS_IGNORE : reinterpret_cast<const MPI_Status *>( status );
}

inline MPI_Datatype *raw_datatypes( mpi_data_type<> *datatypes )
{
    return reinterpret_cast<MPI_Datatype *>( datatypes );
}

inline const MPI_Datatype *raw_datatypes( const mpi_data_type<> *datatypes )
{
    return reinterpret_cast<const MPI_Datatype *>( datatypes );
}

template <class T>
void all_gather( const T *sendbuf, int sendcount, T *recvbuf, int recvcount, MPI_Comm comm )
{
    SCFD_MPI_SAFE_CALL( MPI_Allgather(
        static_cast<const void *>( sendbuf ), sendcount * sizeof( T ), mpi_data_type<char>::mpi_type().native(),
        static_cast<void *>( recvbuf ), recvcount * sizeof( T ), mpi_data_type<char>::mpi_type().native(), comm
    ) );
}


template <class T>
void all_gatherv( const T *sendbuf, int sendcount, T *recvbuf, const int *recvcounts, const int *displs, MPI_Comm comm )
{
    SCFD_MPI_SAFE_CALL( MPI_Allgatherv(
        static_cast<const void *>( sendbuf ), sendcount, mpi_data_type<T>::mpi_type().native(),
        static_cast<void *>( recvbuf ), recvcounts, displs, mpi_data_type<T>::mpi_type().native(), comm
    ) );
}

template <class T>
void gatherv(
    const T *sendbuf, int sendcount, T *recvbuf, const int *recvcounts, const int *displs, int root, MPI_Comm comm
)
{
    SCFD_MPI_SAFE_CALL( MPI_Gatherv(
        static_cast<const void *>( sendbuf ), sendcount, mpi_data_type<T>::mpi_type().native(),
        static_cast<void *>( recvbuf ), recvcounts, displs, mpi_data_type<T>::mpi_type().native(), root, comm
    ) );
}

inline void alltoallv(
    const void *sendbuf, const int *sendcounts, const int *sdispls, const mpi_data_type<> &sendtype, void *recvbuf,
    const int *recvcounts, const int *rdispls, const mpi_data_type<> &recvtype, MPI_Comm comm
)
{
    SCFD_MPI_SAFE_CALL( MPI_Alltoallv(
        sendbuf, sendcounts, sdispls, sendtype.native(), recvbuf, recvcounts, rdispls, recvtype.native(), comm
    ) );
}

template <class T>
void alltoallv(
    const T *sendbuf, const int *sendcounts, const int *sdispls, T *recvbuf, const int *recvcounts,
    const int *rdispls, MPI_Comm comm
)
{
    alltoallv(
        static_cast<const void *>( sendbuf ), sendcounts, sdispls, mpi_data_type<T>::mpi_type(),
        static_cast<void *>( recvbuf ), recvcounts, rdispls, mpi_data_type<T>::mpi_type(), comm
    );
}

inline void alltoallw(
    const void *sendbuf, const int *sendcounts, const int *sdispls, const mpi_data_type<> *sendtypes, void *recvbuf,
    const int *recvcounts, const int *rdispls, const mpi_data_type<> *recvtypes, MPI_Comm comm
)
{
    SCFD_MPI_SAFE_CALL( MPI_Alltoallw(
        sendbuf, sendcounts, sdispls, raw_datatypes( sendtypes ), recvbuf, recvcounts, rdispls,
        raw_datatypes( recvtypes ), comm
    ) );
}

inline void isend(
    const void *buf, int count, const mpi_data_type<> &datatype, int dest, int tag, MPI_Comm comm, mpi_request *request
)
{
    SCFD_MPI_SAFE_CALL( MPI_Isend( buf, count, datatype.native(), dest, tag, comm, raw_requests( request ) ) );
}

template <class T>
void isend( const T *buf, int count, int dest, int tag, MPI_Comm comm, mpi_request *request )
{
    isend( static_cast<const void *>( buf ), count, mpi_data_type<T>::mpi_type(), dest, tag, comm, request );
}

inline void irecv(
    void *buf, int count, const mpi_data_type<> &datatype, int source, int tag, MPI_Comm comm, mpi_request *request
)
{
    SCFD_MPI_SAFE_CALL( MPI_Irecv( buf, count, datatype.native(), source, tag, comm, raw_requests( request ) ) );
}

template <class T>
void irecv( T *buf, int count, int source, int tag, MPI_Comm comm, mpi_request *request )
{
    irecv( static_cast<void *>( buf ), count, mpi_data_type<T>::mpi_type(), source, tag, comm, request );
}

inline void waitall( int count, mpi_request *requests, mpi_status *statuses )
{
    SCFD_MPI_SAFE_CALL( MPI_Waitall( count, raw_requests( requests ), raw_statuses( statuses ) ) );
}

inline void waitall( int count, mpi_request *requests )
{
    waitall( count, requests, static_cast<mpi_status *>( nullptr ) );
}

inline int waitany( int count, mpi_request *requests, mpi_status *status )
{
    int index = MPI_UNDEFINED;
    SCFD_MPI_SAFE_CALL( MPI_Waitany( count, raw_requests( requests ), &index, raw_status( status ) ) );
    return index;
}

inline int waitany( int count, mpi_request *requests )
{
    return waitany( count, requests, static_cast<mpi_status *>( nullptr ) );
}

inline mpi_data_type<> type_vector( int count, int blocklength, int stride, const mpi_data_type<> &oldtype )
{
    mpi_data_type<> newtype;
    SCFD_MPI_SAFE_CALL( MPI_Type_vector( count, blocklength, stride, oldtype.native(), newtype.native_ptr() ) );
    return newtype;
}

inline void type_commit( mpi_data_type<> *datatype )
{
    SCFD_MPI_SAFE_CALL( MPI_Type_commit( datatype->native_ptr() ) );
}

inline void type_commit( mpi_data_type<> &datatype )
{
    type_commit( &datatype );
}

inline void type_free( mpi_data_type<> *datatype )
{
    if ( datatype->is_null() )
        return;
    SCFD_MPI_SAFE_CALL( MPI_Type_free( datatype->native_ptr() ) );
}

inline void type_free( mpi_data_type<> &datatype )
{
    type_free( &datatype );
}

inline double wtime()
{
    return MPI_Wtime();
}


template <class T>
void all_reduce( T *loc_data, T *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    throw std::logic_error( "all_reduce:type not implemented" );
}

template <>
void all_reduce<float>( float *loc_data, float *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<float>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<double>( double *loc_data, double *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<double>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<int>( int *loc_data, int *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<int>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<unsigned int>(
    unsigned int *loc_data, unsigned int *res_data, int count, MPI_Op op, MPI_Comm communicator
)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<unsigned int>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<long>( long *loc_data, long *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<long>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<unsigned long>(
    unsigned long *loc_data, unsigned long *res_data, int count, MPI_Op op, MPI_Comm communicator
)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<unsigned long>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<long long>( long long *loc_data, long long *res_data, int count, MPI_Op op, MPI_Comm communicator )
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<long long>::mpi_type().native(), op, communicator
    ) );
}

template <>
void all_reduce<unsigned long long>(
    unsigned long long *loc_data, unsigned long long *res_data, int count, MPI_Op op, MPI_Comm communicator
)
{
    SCFD_MPI_SAFE_CALL( MPI_Allreduce(
        static_cast<void *>( loc_data ), static_cast<void *>( res_data ), count,
        mpi_data_type<unsigned long long>::mpi_type().native(), op, communicator
    ) );
}

} // namespace detail

class mpi_comm;

struct mpi_comm_info
{
    using mpi_comm_type = mpi_comm;
    using request_type  = detail::mpi_request;
    using status_type   = detail::mpi_status;
    template <class T = void>
    using data_type = detail::mpi_data_type<T>;

    MPI_Comm comm;
    int      num_procs;
    int      myid;
    int      provided_threads;

    void barrier() const
    {
        SCFD_MPI_SAFE_CALL( MPI_Barrier( comm ) );
    }

    template <class T>
    void all_gather( const T *sendbuf, int sendcount, T *recvbuf, int recvcount ) const
    {
        detail::all_gather( sendbuf, sendcount, recvbuf, recvcount, comm );
    }
    template <class T>
    void all_gatherv( const T *sendbuf, int sendcount, T *recvbuf, const int *recvcounts, const int *displs ) const
    {
        detail::all_gatherv( sendbuf, sendcount, recvbuf, recvcounts, displs, comm );
    }
    template <class T>
    void gatherv(
        const T *sendbuf, int sendcount, T *recvbuf, const int *recvcounts, const int *displs, int root
    ) const
    {
        detail::gatherv( sendbuf, sendcount, recvbuf, recvcounts, displs, root, comm );
    }

    void alltoallv(
        const void *sendbuf, const int *sendcounts, const int *sdispls, const detail::mpi_data_type<> &sendtype,
        void *recvbuf, const int *recvcounts, const int *rdispls, const detail::mpi_data_type<> &recvtype
    ) const
    {
        detail::alltoallv( sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm );
    }
    template <class T>
    void alltoallv(
        const T *sendbuf, const int *sendcounts, const int *sdispls, T *recvbuf, const int *recvcounts,
        const int *rdispls
    ) const
    {
        detail::alltoallv( sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, comm );
    }

    void alltoallw(
        const void *sendbuf, const int *sendcounts, const int *sdispls, const detail::mpi_data_type<> *sendtypes,
        void *recvbuf, const int *recvcounts, const int *rdispls, const detail::mpi_data_type<> *recvtypes
    ) const
    {
        detail::alltoallw( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm );
    }

    void isend(
        const void *buf, int count, const detail::mpi_data_type<> &datatype, int dest, int tag,
        detail::mpi_request *request
    ) const
    {
        detail::isend( buf, count, datatype, dest, tag, comm, request );
    }
    void isend(
        const void *buf, int count, const detail::mpi_data_type<> &datatype, int dest, int tag,
        detail::mpi_request &request
    ) const
    {
        isend( buf, count, datatype, dest, tag, &request );
    }
    template <class T>
    void isend( const T *buf, int count, int dest, int tag, detail::mpi_request *request ) const
    {
        detail::isend( buf, count, dest, tag, comm, request );
    }
    template <class T>
    void isend( const T *buf, int count, int dest, int tag, detail::mpi_request &request ) const
    {
        isend( buf, count, dest, tag, &request );
    }

    void irecv(
        void *buf, int count, const detail::mpi_data_type<> &datatype, int source, int tag,
        detail::mpi_request *request
    ) const
    {
        detail::irecv( buf, count, datatype, source, tag, comm, request );
    }
    void irecv(
        void *buf, int count, const detail::mpi_data_type<> &datatype, int source, int tag,
        detail::mpi_request &request
    ) const
    {
        irecv( buf, count, datatype, source, tag, &request );
    }
    template <class T>
    void irecv( T *buf, int count, int source, int tag, detail::mpi_request *request ) const
    {
        detail::irecv( buf, count, source, tag, comm, request );
    }
    template <class T>
    void irecv( T *buf, int count, int source, int tag, detail::mpi_request &request ) const
    {
        irecv( buf, count, source, tag, &request );
    }

    void waitall( int count, detail::mpi_request *requests, detail::mpi_status *statuses ) const
    {
        detail::waitall( count, requests, statuses );
    }
    void waitall( int count, detail::mpi_request *requests ) const
    {
        detail::waitall( count, requests );
    }

    int waitany( int count, detail::mpi_request *requests, detail::mpi_status *status ) const
    {
        return detail::waitany( count, requests, status );
    }
    int waitany( int count, detail::mpi_request *requests ) const
    {
        return detail::waitany( count, requests );
    }

    template <class T>
    void all_reduce( T *loc_data, T *res_data, int count, MPI_Op op ) const
    {
        detail::all_reduce( loc_data, res_data, count, op, comm );
    }
    template <class T>
    void all_reduce_sum( T *loc_data, T *res_data, int count ) const
    {
        all_reduce( loc_data, res_data, count, MPI_SUM );
    }
    template <class T>
    void all_reduce_max( T *loc_data, T *res_data, int count ) const
    {
        all_reduce( loc_data, res_data, count, MPI_MAX );
    }
    template <class T>
    void all_reduce_min( T *loc_data, T *res_data, int count ) const
    {
        all_reduce( loc_data, res_data, count, MPI_MIN );
    }
    /// Shortcuts for 1 value
    template <class T>
    T all_reduce_sum( T loc_val ) const
    {
        T res_val;
        all_reduce_sum( &loc_val, &res_val, 1 );
        return res_val;
    }
    template <class T>
    T all_reduce_max( T loc_val ) const
    {
        T res_val;
        all_reduce_max( &loc_val, &res_val, 1 );
        return res_val;
    }
    template <class T>
    T all_reduce_min( T loc_val ) const
    {
        T res_val;
        all_reduce_min( &loc_val, &res_val, 1 );
        return res_val;
    }

    mpi_comm split( int color, int key ) const;
    mpi_comm split( int color ) const;
    mpi_comm split_type( int color, int key ) const;
    mpi_comm split_type( int color ) const;
};

} // namespace communication
} // namespace scfd


#endif
