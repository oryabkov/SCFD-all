// Copyright (C) 2026 SCFD contributors

#include <cstdint>
#include <cstring>
#include <vector>

#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

namespace
{

enum error_code_t
{
    test_success                              = 0,
    error_bcast_int                           = 1,
    error_bcast_double                        = 2,
    error_bcast_unsigned_char                 = 3,
    error_bcast_uint8_t                       = 4,
    error_all_gatherv_unsigned_char           = 5,
    error_all_gatherv_uint8_t                 = 6,
    error_iall_gather_int_status              = 7,
    error_iall_gather_int_no_status           = 8,
    error_iall_gather_double_status           = 9,
    error_iall_gather_double_no_status        = 10,
    error_iall_gather_unsigned_char_status    = 11,
    error_iall_gather_unsigned_char_no_status = 12,
    error_iall_gather_uint8_t_status          = 13,
    error_iall_gather_uint8_t_no_status       = 14,
    error_raw_bcast_datatype                  = 15,
    error_bcast_bytes                         = 16,
    error_all_gatherv_bytes                   = 17,
    error_wait_recv_value                     = 18,
    error_wait_status_source                  = 19,
    error_wait_status_tag                     = 20,
    error_all_gather_int                      = 21,
    error_all_gatherv_long_counts             = 22,
    error_type_contiguous_raw_alltoallv       = 23,
    error_type_contiguous_free                = 24,
    error_all_reduce_sum_array                = 25,
    error_all_reduce_max_array                = 26,
    error_all_reduce_min_array                = 27,
    error_all_reduce_sum_scalar               = 28,
    error_all_reduce_max_scalar               = 29,
    error_all_reduce_min_scalar               = 30,
    error_split                               = 31,
    error_split_type                          = 32,
    error_all_reduce_direct                   = 33
};

const char *error_code_name( int code )
{
    switch ( code )
    {
    case test_success:
        return "success";
    case error_bcast_int:
        return "bcast<int>";
    case error_bcast_double:
        return "bcast<double>";
    case error_bcast_unsigned_char:
        return "bcast<unsigned char>";
    case error_bcast_uint8_t:
        return "bcast<std::uint8_t>";
    case error_all_gatherv_unsigned_char:
        return "all_gatherv<unsigned char>";
    case error_all_gatherv_uint8_t:
        return "all_gatherv<std::uint8_t>";
    case error_iall_gather_int_status:
        return "iall_gather<int> with wait status";
    case error_iall_gather_int_no_status:
        return "iall_gather<int> with no-status wait";
    case error_iall_gather_double_status:
        return "iall_gather<double> with wait status";
    case error_iall_gather_double_no_status:
        return "iall_gather<double> with no-status wait";
    case error_iall_gather_unsigned_char_status:
        return "iall_gather<unsigned char> with wait status";
    case error_iall_gather_unsigned_char_no_status:
        return "iall_gather<unsigned char> with no-status wait";
    case error_iall_gather_uint8_t_status:
        return "iall_gather<std::uint8_t> with wait status";
    case error_iall_gather_uint8_t_no_status:
        return "iall_gather<std::uint8_t> with no-status wait";
    case error_raw_bcast_datatype:
        return "raw datatype bcast";
    case error_bcast_bytes:
        return "bcast_bytes";
    case error_all_gatherv_bytes:
        return "all_gatherv_bytes";
    case error_wait_recv_value:
        return "wait receive value";
    case error_wait_status_source:
        return "wait status source";
    case error_wait_status_tag:
        return "wait status tag";
    case error_all_gather_int:
        return "all_gather<int>";
    case error_all_gatherv_long_counts:
        return "all_gatherv<int> with long counts";
    case error_type_contiguous_raw_alltoallv:
        return "type_contiguous with raw alltoallv";
    case error_type_contiguous_free:
        return "type_contiguous type_free";
    case error_all_reduce_sum_array:
        return "all_reduce_sum array";
    case error_all_reduce_max_array:
        return "all_reduce_max array";
    case error_all_reduce_min_array:
        return "all_reduce_min array";
    case error_all_reduce_sum_scalar:
        return "all_reduce_sum scalar";
    case error_all_reduce_max_scalar:
        return "all_reduce_max scalar";
    case error_all_reduce_min_scalar:
        return "all_reduce_min scalar";
    case error_split:
        return "split";
    case error_split_type:
        return "split_type";
    case error_all_reduce_direct:
        return "all_reduce direct";
    default:
        return "unknown";
    }
}

struct raw_payload_t
{
    int           rank;
    int           item;
    double        value;
    unsigned char bytes[4];
};

raw_payload_t make_payload( int rank, int item )
{
    raw_payload_t payload;
    payload.rank     = rank;
    payload.item     = item;
    payload.value    = 1000.0 * static_cast<double>( rank ) + static_cast<double>( item );
    payload.bytes[0] = static_cast<unsigned char>( rank + item );
    payload.bytes[1] = static_cast<unsigned char>( 2 * rank + item );
    payload.bytes[2] = static_cast<unsigned char>( rank + 2 * item );
    payload.bytes[3] = static_cast<unsigned char>( 3 * rank + 5 * item );
    return payload;
}

bool payload_equal( const raw_payload_t &left, const raw_payload_t &right )
{
    return ( left.rank == right.rank ) && ( left.item == right.item ) && ( left.value == right.value ) &&
           ( left.bytes[0] == right.bytes[0] ) && ( left.bytes[1] == right.bytes[1] ) &&
           ( left.bytes[2] == right.bytes[2] ) && ( left.bytes[3] == right.bytes[3] );
}

template <class T>
T make_scalar_value( int rank, int item )
{
    return static_cast<T>( 17 * rank + 5 * item + 3 );
}

template <class T, class CommInfo, class Fail>
int test_typed_bcast( const CommInfo &comm_info, const char *type_name, int error_code, Fail fail )
{
    int result = test_success;

    for ( int root = 0; root < comm_info.num_procs; ++root )
    {
        T value = comm_info.myid == root ? make_scalar_value<T>( root, 0 ) : T();
        comm_info.bcast( &value, 1, root );

        const T expected = make_scalar_value<T>( root, 0 );
        if ( value != expected )
        {
            if ( result == test_success )
            {
                result = error_code;
                fail( error_code, "bcast<%s> failed on rank %d with root %d", type_name, comm_info.myid, root );
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_all_gather_int( const CommInfo &comm_info, Fail fail )
{
    const int fixed_count = 3;

    std::vector<int> send( fixed_count );
    for ( int i = 0; i < fixed_count; ++i )
    {
        send[i] = make_scalar_value<int>( comm_info.myid, i );
    }

    std::vector<int> recv( comm_info.num_procs * fixed_count, -1 );
    comm_info.all_gather( send.data(), fixed_count, recv.data(), fixed_count );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( int i = 0; i < fixed_count; ++i )
        {
            const int expected = make_scalar_value<int>( rank, i );
            const int actual   = recv[rank * fixed_count + i];
            if ( actual != expected )
            {
                fail(
                    error_all_gather_int, "all_gather<int> failed on rank %d for source rank %d item %d",
                    comm_info.myid, rank, i
                );
                return error_all_gather_int;
            }
        }
    }

    return test_success;
}

template <class T, class CommInfo, class Fail>
int test_typed_all_gatherv( const CommInfo &comm_info, const char *type_name, int error_code, Fail fail )
{
    int       result      = test_success;
    const int local_count = comm_info.myid + 2;

    std::vector<T> send( local_count );
    for ( int i = 0; i < local_count; ++i )
    {
        send[i] = make_scalar_value<T>( comm_info.myid, i );
    }

    std::vector<int> recvcounts( comm_info.num_procs );
    std::vector<int> displs( comm_info.num_procs );
    int              total_count = 0;
    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        recvcounts[rank] = rank + 2;
        displs[rank]     = total_count;
        total_count += recvcounts[rank];
    }

    std::vector<T> recv( total_count, T() );
    comm_info.all_gatherv( send.data(), local_count, recv.data(), recvcounts.data(), displs.data() );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( int i = 0; i < recvcounts[rank]; ++i )
        {
            const T expected = make_scalar_value<T>( rank, i );
            const T actual   = recv[displs[rank] + i];
            if ( actual != expected )
            {
                if ( result == test_success )
                {
                    result = error_code;
                    fail(
                        error_code, "all_gatherv<%s> failed on rank %d for source rank %d item %d", type_name,
                        comm_info.myid, rank, i
                    );
                }
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_all_gatherv_long_counts( const CommInfo &comm_info, Fail fail )
{
    const long local_count = static_cast<long>( comm_info.myid + 1 );

    std::vector<int> send( static_cast<std::size_t>( local_count ) );
    for ( long i = 0; i < local_count; ++i )
    {
        send[static_cast<std::size_t>( i )] = make_scalar_value<int>( comm_info.myid, static_cast<int>( i ) );
    }

    std::vector<long> recvcounts( comm_info.num_procs );
    std::vector<long> displs( comm_info.num_procs );
    long              total_count = 0;
    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        recvcounts[rank] = static_cast<long>( rank + 1 );
        displs[rank]     = total_count;
        total_count += recvcounts[rank];
    }

    std::vector<int> recv( static_cast<std::size_t>( total_count ), -1 );
    comm_info.all_gatherv( send.data(), local_count, recv.data(), recvcounts.data(), displs.data() );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( long i = 0; i < recvcounts[rank]; ++i )
        {
            const int expected = make_scalar_value<int>( rank, static_cast<int>( i ) );
            const int actual   = recv[static_cast<std::size_t>( displs[rank] + i )];
            if ( actual != expected )
            {
                fail(
                    error_all_gatherv_long_counts,
                    "all_gatherv<int,long> failed on rank %d for source rank %d item %ld", comm_info.myid, rank, i
                );
                return error_all_gatherv_long_counts;
            }
        }
    }

    return test_success;
}

template <class T, class CommInfo, class Fail>
int test_iall_gather(
    const CommInfo &comm_info, const char *type_name, int status_error, int no_status_error, Fail fail
)
{
    using request_t = typename CommInfo::request_type;
    using status_t  = typename CommInfo::status_type;

    int       result      = test_success;
    const int fixed_count = 4;

    std::vector<T> send( fixed_count );
    for ( int i = 0; i < fixed_count; ++i )
    {
        send[i] = make_scalar_value<T>( comm_info.myid, i );
    }

    std::vector<T> recv( comm_info.num_procs * fixed_count, T() );
    request_t      request;
    status_t       status;

    comm_info.iall_gather( send.data(), fixed_count, recv.data(), fixed_count, &request );
    comm_info.wait( &request, &status );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( int i = 0; i < fixed_count; ++i )
        {
            const T expected = make_scalar_value<T>( rank, i );
            const T actual   = recv[rank * fixed_count + i];
            if ( actual != expected )
            {
                if ( result == test_success )
                {
                    result = status_error;
                    fail(
                        status_error, "iall_gather<%s> failed on rank %d for source rank %d item %d", type_name,
                        comm_info.myid, rank, i
                    );
                }
            }
        }
    }

    for ( int i = 0; i < fixed_count; ++i )
    {
        send[i] = make_scalar_value<T>( comm_info.myid, i + fixed_count );
        recv[i] = T();
    }

    comm_info.iall_gather( send.data(), fixed_count, recv.data(), fixed_count, request );
    comm_info.wait( request );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( int i = 0; i < fixed_count; ++i )
        {
            const T expected = make_scalar_value<T>( rank, i + fixed_count );
            const T actual   = recv[rank * fixed_count + i];
            if ( actual != expected )
            {
                if ( result == test_success )
                {
                    result = no_status_error;
                    fail(
                        no_status_error, "iall_gather<%s> no-status wait failed on rank %d for source rank %d item %d",
                        type_name, comm_info.myid, rank, i
                    );
                }
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_raw_bcast_datatype( const CommInfo &comm_info, Fail fail )
{
    int result = test_success;

    for ( int root = 0; root < comm_info.num_procs; ++root )
    {
        int value = comm_info.myid == root ? make_scalar_value<int>( root, 13 ) : -1;
        comm_info.bcast( static_cast<void *>( &value ), 1, comm_info.template get_data_type<int>(), root );

        const int expected = make_scalar_value<int>( root, 13 );
        if ( value != expected )
        {
            if ( result == test_success )
            {
                result = error_raw_bcast_datatype;
                fail(
                    error_raw_bcast_datatype, "raw datatype bcast failed on rank %d with root %d", comm_info.myid, root
                );
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_bcast_bytes( const CommInfo &comm_info, Fail fail )
{
    int result = test_success;

    for ( int root = 0; root < comm_info.num_procs; ++root )
    {
        raw_payload_t payload = comm_info.myid == root ? make_payload( root, 7 ) : make_payload( -1, -1 );
        comm_info.bcast_bytes( &payload, static_cast<int>( sizeof( payload ) ), root );

        const raw_payload_t expected = make_payload( root, 7 );
        if ( !payload_equal( payload, expected ) )
        {
            if ( result == test_success )
            {
                result = error_bcast_bytes;
                fail( error_bcast_bytes, "bcast_bytes failed on rank %d with root %d", comm_info.myid, root );
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_all_gatherv_bytes( const CommInfo &comm_info, Fail fail )
{
    int       result      = test_success;
    const int local_count = comm_info.myid + 1;

    std::vector<raw_payload_t> send( local_count );
    for ( int i = 0; i < local_count; ++i )
    {
        send[i] = make_payload( comm_info.myid, i );
    }

    std::vector<int> recvcounts( comm_info.num_procs );
    std::vector<int> displs( comm_info.num_procs );
    int              total_bytes = 0;
    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        recvcounts[rank] = static_cast<int>( sizeof( raw_payload_t ) ) * ( rank + 1 );
        displs[rank]     = total_bytes;
        total_bytes += recvcounts[rank];
    }

    std::vector<unsigned char> recv( total_bytes, 0 );
    comm_info.all_gatherv_bytes(
        send.data(), static_cast<int>( sizeof( raw_payload_t ) ) * local_count, recv.data(), recvcounts.data(),
        displs.data()
    );

    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        for ( int i = 0; i < rank + 1; ++i )
        {
            raw_payload_t actual;
            const int     byte_offset = displs[rank] + static_cast<int>( sizeof( raw_payload_t ) ) * i;
            std::memcpy( &actual, recv.data() + byte_offset, sizeof( actual ) );

            const raw_payload_t expected = make_payload( rank, i );
            if ( !payload_equal( actual, expected ) )
            {
                if ( result == test_success )
                {
                    result = error_all_gatherv_bytes;
                    fail(
                        error_all_gatherv_bytes, "all_gatherv_bytes failed on rank %d for source rank %d item %d",
                        comm_info.myid, rank, i
                    );
                }
            }
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_type_contiguous_raw_alltoallv( const CommInfo &comm_info, Fail fail )
{
    using mpi_dtype_t = typename CommInfo::data_type;

    const int pair_size = 2;

    std::vector<int> send( static_cast<std::size_t>( comm_info.num_procs * pair_size ) );
    std::vector<int> recv( static_cast<std::size_t>( comm_info.num_procs * pair_size ), -1 );

    for ( int dest = 0; dest < comm_info.num_procs; ++dest )
    {
        send[static_cast<std::size_t>( pair_size * dest )]     = 100 * comm_info.myid + dest;
        send[static_cast<std::size_t>( pair_size * dest + 1 )] = 1000 * comm_info.myid + 10 * dest + 1;
    }

    std::vector<int> sendcounts( comm_info.num_procs, 1 );
    std::vector<int> recvcounts( comm_info.num_procs, 1 );
    std::vector<int> displs( comm_info.num_procs );
    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        displs[rank] = rank;
    }

    mpi_dtype_t pair_type = comm_info.type_contiguous( pair_size, comm_info.template get_data_type<int>() );
    comm_info.type_commit( &pair_type );

    comm_info.alltoallv(
        static_cast<const void *>( send.data() ), sendcounts.data(), displs.data(), pair_type,
        static_cast<void *>( recv.data() ), recvcounts.data(), displs.data(), pair_type
    );

    int result = test_success;
    for ( int src = 0; src < comm_info.num_procs; ++src )
    {
        const int expected0 = 100 * src + comm_info.myid;
        const int expected1 = 1000 * src + 10 * comm_info.myid + 1;
        const int actual0   = recv[static_cast<std::size_t>( pair_size * src )];
        const int actual1   = recv[static_cast<std::size_t>( pair_size * src + 1 )];
        if ( ( actual0 != expected0 ) || ( actual1 != expected1 ) )
        {
            result = error_type_contiguous_raw_alltoallv;
            fail(
                error_type_contiguous_raw_alltoallv, "type_contiguous raw alltoallv failed on rank %d from src %d",
                comm_info.myid, src
            );
            break;
        }
    }

    comm_info.type_free( &pair_type );
    if ( ( result == test_success ) && !pair_type.is_null() )
    {
        fail( error_type_contiguous_free, "type_free pointer overload did not reset contiguous datatype" );
        return error_type_contiguous_free;
    }

    return result;
}

template <class CommInfo, class Fail>
int test_all_reduce_variants( const CommInfo &comm_info, Fail fail )
{
    int loc[3] = { comm_info.myid + 1, 2 * comm_info.myid - 3, 7 - comm_info.myid };
    int res[3] = { 0, 0, 0 };
    int result = test_success;

    comm_info.all_reduce_sum( loc, res, 3 );

    int expected_sum[3] = { 0, 0, 0 };
    int expected_max[3] = { loc[0], loc[1], loc[2] };
    int expected_min[3] = { loc[0], loc[1], loc[2] };
    for ( int rank = 0; rank < comm_info.num_procs; ++rank )
    {
        const int values[3] = { rank + 1, 2 * rank - 3, 7 - rank };
        for ( int i = 0; i < 3; ++i )
        {
            expected_sum[i] += values[i];
            if ( values[i] > expected_max[i] )
                expected_max[i] = values[i];
            if ( values[i] < expected_min[i] )
                expected_min[i] = values[i];
        }
    }

    for ( int i = 0; i < 3; ++i )
    {
        if ( res[i] != expected_sum[i] )
        {
            if ( result == test_success )
            {
                result = error_all_reduce_sum_array;
                fail( error_all_reduce_sum_array, "all_reduce_sum array failed on rank %d", comm_info.myid );
            }
        }
    }

    comm_info.all_reduce_max( loc, res, 3 );
    for ( int i = 0; i < 3; ++i )
    {
        if ( res[i] != expected_max[i] )
        {
            if ( result == test_success )
            {
                result = error_all_reduce_max_array;
                fail( error_all_reduce_max_array, "all_reduce_max array failed on rank %d", comm_info.myid );
            }
        }
    }

    comm_info.all_reduce_min( loc, res, 3 );
    for ( int i = 0; i < 3; ++i )
    {
        if ( res[i] != expected_min[i] )
        {
            if ( result == test_success )
            {
                result = error_all_reduce_min_array;
                fail( error_all_reduce_min_array, "all_reduce_min array failed on rank %d", comm_info.myid );
            }
        }
    }

    comm_info.all_reduce( loc, res, 3, MPI_SUM );
    for ( int i = 0; i < 3; ++i )
    {
        if ( res[i] != expected_sum[i] )
        {
            if ( result == test_success )
            {
                result = error_all_reduce_direct;
                fail( error_all_reduce_direct, "all_reduce direct failed on rank %d", comm_info.myid );
            }
        }
    }

    if ( comm_info.all_reduce_sum( loc[0] ) != expected_sum[0] )
    {
        if ( result == test_success )
        {
            result = error_all_reduce_sum_scalar;
            fail( error_all_reduce_sum_scalar, "all_reduce_sum scalar failed on rank %d", comm_info.myid );
        }
    }
    if ( comm_info.all_reduce_max( loc[1] ) != expected_max[1] )
    {
        if ( result == test_success )
        {
            result = error_all_reduce_max_scalar;
            fail( error_all_reduce_max_scalar, "all_reduce_max scalar failed on rank %d", comm_info.myid );
        }
    }
    if ( comm_info.all_reduce_min( loc[2] ) != expected_min[2] )
    {
        if ( result == test_success )
        {
            result = error_all_reduce_min_scalar;
            fail( error_all_reduce_min_scalar, "all_reduce_min scalar failed on rank %d", comm_info.myid );
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_split_variants( const CommInfo &comm_info, Fail fail )
{
    const int color  = comm_info.myid % 2;
    int       result = test_success;

    auto split_comm = comm_info.split( color );
    auto split_info = split_comm.info();

    const int split_count          = split_info.all_reduce_sum( 1 );
    const int expected_split_count = ( comm_info.num_procs + ( color == 0 ? 1 : 0 ) ) / 2;
    if ( split_count != expected_split_count )
    {
        result = error_split;
        fail( error_split, "split failed on rank %d", comm_info.myid );
    }

    auto      shared_comm  = comm_info.split_type( MPI_COMM_TYPE_SHARED );
    auto      shared_info  = shared_comm.info();
    const int shared_count = shared_info.all_reduce_sum( 1 );
    if ( ( shared_count <= 0 ) || ( shared_count > comm_info.num_procs ) )
    {
        if ( result == test_success )
        {
            result = error_split_type;
            fail( error_split_type, "split_type failed on rank %d", comm_info.myid );
        }
    }

    return result;
}

template <class CommInfo, class Fail>
int test_wait_single_request( const CommInfo &comm_info, Fail fail )
{
    using request_t = typename CommInfo::request_type;
    using status_t  = typename CommInfo::status_type;

    const int left  = ( comm_info.myid + comm_info.num_procs - 1 ) % comm_info.num_procs;
    const int right = ( comm_info.myid + 1 ) % comm_info.num_procs;

    int       send_value = make_scalar_value<int>( comm_info.myid, 11 );
    int       recv_value = -1;
    request_t recv_request;
    request_t send_request;
    status_t  recv_status;

    comm_info.irecv( &recv_value, 1, left, 701, &recv_request );
    comm_info.isend( &send_value, 1, right, 701, &send_request );

    comm_info.wait( recv_request, &recv_status );
    comm_info.wait( &send_request );

    const int expected = make_scalar_value<int>( left, 11 );
    if ( recv_value != expected )
    {
        fail( error_wait_recv_value, "wait failed to complete receive on rank %d", comm_info.myid );
        return error_wait_recv_value;
    }
    if ( recv_status.source() != left )
    {
        fail( error_wait_status_source, "wait returned wrong source on rank %d", comm_info.myid );
        return error_wait_status_source;
    }
    if ( recv_status.tag() != 701 )
    {
        fail( error_wait_status_tag, "wait returned wrong tag on rank %d", comm_info.myid );
        return error_wait_status_tag;
    }

    return test_success;
}

template <class CommInfo, class Log, class Test>
int run_test_case( const CommInfo &comm_info, Log &log, const char *test_name, Test test )
{
    const int local_error  = test();
    const int global_error = comm_info.all_reduce_max( local_error );

    if ( global_error != test_success )
    {
        if ( comm_info.myid == 0 )
        {
            log.error_f(
                "FAILED: code=%d, test=%s, call=%s", global_error, test_name, error_code_name( global_error )
            );
        }
        return global_error;
    }

    return test_success;
}

} // namespace

int main( int argc, char *argv[] )
{
    using mpi_wrap_t = scfd::communication::mpi_wrap;
    using log_t      = scfd::utils::log_mpi;

    mpi_wrap_t mpi( argc, argv );
    auto       comm_info = mpi.comm_world();
    log_t      log;

    auto fail = [&]( int code, const char *fmt, auto... args ) {
        log.error_f( "code=%d, call=%s", code, error_code_name( code ) );
        log.error_f( fmt, args... );
    };

    int failed_global = test_success;

    failed_global = run_test_case( comm_info, log, "typed bcast<int>", [&]() {
        return test_typed_bcast<int>( comm_info, "int", error_bcast_int, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "typed bcast<double>", [&]() {
        return test_typed_bcast<double>( comm_info, "double", error_bcast_double, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "typed bcast<unsigned char>", [&]() {
        return test_typed_bcast<unsigned char>( comm_info, "unsigned char", error_bcast_unsigned_char, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "typed bcast<std::uint8_t>", [&]() {
        return test_typed_bcast<std::uint8_t>( comm_info, "std::uint8_t", error_bcast_uint8_t, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global =
        run_test_case( comm_info, log, "all_gather<int>", [&]() { return test_all_gather_int( comm_info, fail ); } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "typed all_gatherv<unsigned char>", [&]() {
        return test_typed_all_gatherv<unsigned char>(
            comm_info, "unsigned char", error_all_gatherv_unsigned_char, fail
        );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "typed all_gatherv<std::uint8_t>", [&]() {
        return test_typed_all_gatherv<std::uint8_t>( comm_info, "std::uint8_t", error_all_gatherv_uint8_t, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "all_gatherv<int> with long counts", [&]() {
        return test_all_gatherv_long_counts( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "iall_gather<int>", [&]() {
        return test_iall_gather<int>(
            comm_info, "int", error_iall_gather_int_status, error_iall_gather_int_no_status, fail
        );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "iall_gather<double>", [&]() {
        return test_iall_gather<double>(
            comm_info, "double", error_iall_gather_double_status, error_iall_gather_double_no_status, fail
        );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "iall_gather<unsigned char>", [&]() {
        return test_iall_gather<unsigned char>(
            comm_info, "unsigned char", error_iall_gather_unsigned_char_status,
            error_iall_gather_unsigned_char_no_status, fail
        );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "iall_gather<std::uint8_t>", [&]() {
        return test_iall_gather<std::uint8_t>(
            comm_info, "std::uint8_t", error_iall_gather_uint8_t_status, error_iall_gather_uint8_t_no_status, fail
        );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "raw datatype bcast", [&]() {
        return test_raw_bcast_datatype( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global =
        run_test_case( comm_info, log, "bcast_bytes", [&]() { return test_bcast_bytes( comm_info, fail ); } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "all_gatherv_bytes", [&]() {
        return test_all_gatherv_bytes( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "type_contiguous raw alltoallv", [&]() {
        return test_type_contiguous_raw_alltoallv( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "all_reduce variants", [&]() {
        return test_all_reduce_variants( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global =
        run_test_case( comm_info, log, "split variants", [&]() { return test_split_variants( comm_info, fail ); } );
    if ( failed_global != test_success )
        return failed_global;

    failed_global = run_test_case( comm_info, log, "wait single request", [&]() {
        return test_wait_single_request( comm_info, fail );
    } );
    if ( failed_global != test_success )
        return failed_global;

    if ( comm_info.myid == 0 )
    {
        log.info( "PASSED" );
    }

    return test_success;
}
