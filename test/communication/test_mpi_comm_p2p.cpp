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

#include <array>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/memory/host.h>
#include <scfd/utils/log_mpi.h>

namespace
{

using value_t     = int;
using memory_t    = scfd::memory::host;
using tensor_2d_t = scfd::arrays::array_nd<value_t, 2, memory_t>;

value_t ring_value( int rank, int message_id, int j )
{
    return 1000 * rank + 100 * message_id + j;
}

} // namespace

int main( int argc, char *argv[] )
{
    using mpi_wrap_t = scfd::communication::mpi_wrap;
    using log_t      = scfd::utils::log_mpi;

    mpi_wrap_t mpi( argc, argv );
    auto       comm_info = mpi.comm_world();
    log_t      log;

    const int num_procs = comm_info.num_procs;
    const int myid      = comm_info.myid;
    const int left      = ( myid + num_procs - 1 ) % num_procs;
    const int right     = ( myid + 1 ) % num_procs;

    bool is_failed = false;

    auto fail = [&]( const char *fmt, auto... args )
    {
        log.error_f( fmt, args... );
        is_failed = true;
    };

    const int payload_size = 4;

    tensor_2d_t send_tensor;
    tensor_2d_t recv_tensor;
    send_tensor.init( 2, payload_size );
    recv_tensor.init( 2, payload_size );

    for ( int message_id = 0; message_id < 2; ++message_id )
    {
        for ( int j = 0; j < payload_size; ++j )
        {
            send_tensor( message_id, j ) = ring_value( myid, message_id, j );
            recv_tensor( message_id, j ) = -1;
        }
    }

    std::vector<MPI_Request> recv_requests( 2, MPI_REQUEST_NULL );
    std::vector<MPI_Request> send_requests( 2, MPI_REQUEST_NULL );

    comm_info.irecv( recv_tensor.raw_ptr(), payload_size, left, 100, recv_requests[0] );
    comm_info.irecv(
        static_cast<void *>( recv_tensor.raw_ptr() + payload_size ), payload_size * static_cast<int>( sizeof( value_t ) ),
        MPI_BYTE, left, 101, recv_requests[1]
    );

    comm_info.isend( send_tensor.raw_ptr(), payload_size, right, 100, send_requests[0] );
    comm_info.isend(
        static_cast<const void *>( send_tensor.raw_ptr() + payload_size ),
        payload_size * static_cast<int>( sizeof( value_t ) ), MPI_BYTE, right, 101, send_requests[1]
    );

    std::array<int, 2> recv_seen{ 0, 0 };
    for ( int iter = 0; iter < 2; ++iter )
    {
        MPI_Status status;
        const int  idx = comm_info.waitany( 2, recv_requests.data(), &status );
        if ( ( idx < 0 ) || ( idx > 1 ) )
        {
            fail( "MPI_Waitany returned invalid index %d on rank %d", idx, myid );
            continue;
        }
        if ( recv_seen[idx] != 0 )
        {
            fail( "MPI_Waitany returned the same completed request %d twice on rank %d", idx, myid );
        }
        recv_seen[idx] = 1;
        if ( status.MPI_SOURCE != left )
        {
            fail(
                "MPI_Waitany returned unexpected source on rank %d: got %d, expected %d", myid, status.MPI_SOURCE,
                left
            );
        }
        if ( status.MPI_TAG != 100 + idx )
        {
            fail( "MPI_Waitany returned unexpected tag on rank %d: got %d, expected %d", myid, status.MPI_TAG, 100 + idx );
        }
    }

    std::vector<MPI_Status> send_statuses( 2 );
    comm_info.waitall( 2, send_requests.data(), send_statuses.data() );

    for ( int message_id = 0; message_id < 2; ++message_id )
    {
        for ( int j = 0; j < payload_size; ++j )
        {
            const value_t expected = ring_value( left, message_id, j );
            if ( recv_tensor( message_id, j ) != expected )
            {
                fail(
                    "ring exchange failed on rank %d for message %d, j=%d: got %d, expected %d", myid, message_id, j,
                    recv_tensor( message_id, j ), expected
                );
            }
        }
    }

    comm_info.barrier();
    const double t0 = scfd::communication::detail::wtime();
    comm_info.barrier();
    const double t1 = scfd::communication::detail::wtime();
    if ( t1 < t0 )
    {
        fail( "MPI_Wtime is not monotone on rank %d: t0=%lf, t1=%lf", myid, t0, t1 );
    }

    std::array<value_t, 2> send_small{ myid, myid + 100 };
    std::array<value_t, 2> recv_small{ -1, -1 };
    recv_requests.assign( 2, MPI_REQUEST_NULL );
    send_requests.assign( 2, MPI_REQUEST_NULL );

    comm_info.irecv( &recv_small[0], 1, left, 200, recv_requests[0] );
    comm_info.irecv( &recv_small[1], 1, left, 201, recv_requests[1] );
    comm_info.isend( &send_small[0], 1, right, 200, send_requests[0] );
    comm_info.isend( &send_small[1], 1, right, 201, send_requests[1] );

    std::array<int, 2> recv_small_seen{ 0, 0 };
    for ( int iter = 0; iter < 2; ++iter )
    {
        const int idx = comm_info.waitany( 2, recv_requests.data() );
        if ( ( idx < 0 ) || ( idx > 1 ) )
        {
            fail( "MPI_Waitany(no status) returned invalid index %d on rank %d", idx, myid );
            continue;
        }
        recv_small_seen[idx] = 1;
    }

    comm_info.waitall( 2, send_requests.data() );

    if ( recv_small[0] != left )
    {
        fail( "second ring exchange failed for tag 200 on rank %d: got %d, expected %d", myid, recv_small[0], left );
    }
    if ( recv_small[1] != left + 100 )
    {
        fail(
            "second ring exchange failed for tag 201 on rank %d: got %d, expected %d", myid, recv_small[1], left + 100
        );
    }

    const int failed_global = comm_info.all_reduce_max( is_failed ? 1 : 0 );
    if ( myid == 0 )
    {
        if ( failed_global == 0 )
            log.info( "PASSED" );
        else
            log.error( "FAILED" );
    }

    return failed_global == 0 ? 0 : 1;
}
