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

#include <numeric>
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

value_t gather_value( int rank, int i, int j )
{
    return 100 * rank + 10 * i + j;
}

value_t alltoallv_value( int source_rank, int destination_rank, int j )
{
    return 1000 * source_rank + 100 * destination_rank + j;
}

value_t alltoallw_value( int source_rank, int row, int column )
{
    return 1000 * source_rank + 100 * row + column;
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

    bool is_failed = false;

    auto fail = [&]( const char *fmt, auto... args )
    {
        log.error_f( fmt, args... );
        is_failed = true;
    };

    // Test MPI_Gatherv through a simple rank-dependent tensor partition.
    const int gather_cols = 3;
    const int gather_rows = myid + 1;

    tensor_2d_t gather_local;
    gather_local.init( gather_rows, gather_cols );
    for ( int i = 0; i < gather_rows; ++i )
    {
        for ( int j = 0; j < gather_cols; ++j )
        {
            gather_local( i, j ) = gather_value( myid, i, j );
        }
    }

    std::vector<int> gatherv_counts( num_procs );
    std::vector<int> gatherv_displs( num_procs );
    int              gathered_total_size = 0;
    for ( int rank = 0; rank < num_procs; ++rank )
    {
        gatherv_counts[rank] = ( rank + 1 ) * gather_cols;
        gatherv_displs[rank] = gathered_total_size;
        gathered_total_size += gatherv_counts[rank];
    }

    std::vector<value_t> gathered( gathered_total_size, -1 );
    comm_info.gatherv(
        gather_local.raw_ptr(), gather_rows * gather_cols, gathered.data(), gatherv_counts.data(), gatherv_displs.data(),
        0
    );

    if ( myid == 0 )
    {
        for ( int rank = 0; rank < num_procs; ++rank )
        {
            for ( int i = 0; i < rank + 1; ++i )
            {
                for ( int j = 0; j < gather_cols; ++j )
                {
                    const int     idx      = gatherv_displs[rank] + i * gather_cols + j;
                    const value_t expected = gather_value( rank, i, j );
                    if ( gathered[idx] != expected )
                    {
                        fail(
                            "MPI_Gatherv failed at rank block %d, i=%d, j=%d: got %d, expected %d", rank, i, j,
                            gathered[idx], expected
                        );
                    }
                }
            }
        }
    }

    // Test MPI_Alltoallv with one contiguous tensor slice per destination rank.
    const int exchange_cols = 3;

    tensor_2d_t alltoallv_send;
    tensor_2d_t alltoallv_recv;
    alltoallv_send.init( num_procs, exchange_cols );
    alltoallv_recv.init( num_procs, exchange_cols );

    for ( int dest = 0; dest < num_procs; ++dest )
    {
        for ( int j = 0; j < exchange_cols; ++j )
        {
            alltoallv_send( dest, j ) = alltoallv_value( myid, dest, j );
            alltoallv_recv( dest, j ) = -1;
        }
    }

    std::vector<int> alltoallv_counts( num_procs, exchange_cols );
    std::vector<int> alltoallv_displs( num_procs );
    for ( int rank = 0; rank < num_procs; ++rank )
    {
        alltoallv_displs[rank] = rank * exchange_cols;
    }

    comm_info.alltoallv(
        alltoallv_send.raw_ptr(), alltoallv_counts.data(), alltoallv_displs.data(), alltoallv_recv.raw_ptr(),
        alltoallv_counts.data(), alltoallv_displs.data()
    );

    for ( int src = 0; src < num_procs; ++src )
    {
        for ( int j = 0; j < exchange_cols; ++j )
        {
            const value_t expected = alltoallv_value( src, myid, j );
            if ( alltoallv_recv( src, j ) != expected )
            {
                fail(
                    "MPI_Alltoallv failed for src=%d, j=%d on rank %d: got %d, expected %d", src, j, myid,
                    alltoallv_recv( src, j ), expected
                );
            }
        }
    }

    // Test MPI_Type_vector + MPI_Type_commit/free + MPI_Alltoallw with a column exchange.
    tensor_2d_t alltoallw_send;
    tensor_2d_t alltoallw_recv;
    alltoallw_send.init( num_procs, num_procs );
    alltoallw_recv.init( num_procs, num_procs );

    for ( int row = 0; row < num_procs; ++row )
    {
        for ( int col = 0; col < num_procs; ++col )
        {
            alltoallw_send( row, col ) = alltoallw_value( myid, row, col );
            alltoallw_recv( row, col ) = -1;
        }
    }

    MPI_Datatype column_type = scfd::communication::detail::type_vector( num_procs, 1, num_procs, MPI_INT );
    scfd::communication::detail::type_commit( column_type );

    std::vector<int>          alltoallw_sendcounts( num_procs, 1 );
    std::vector<int>          alltoallw_sdispls( num_procs );
    std::vector<MPI_Datatype> alltoallw_sendtypes( num_procs, column_type );
    std::vector<int>          alltoallw_recvcounts( num_procs, num_procs );
    std::vector<int>          alltoallw_rdispls( num_procs );
    std::vector<MPI_Datatype> alltoallw_recvtypes( num_procs, MPI_INT );

    for ( int rank = 0; rank < num_procs; ++rank )
    {
        alltoallw_sdispls[rank] = rank * static_cast<int>( sizeof( value_t ) );
        alltoallw_rdispls[rank] = rank * num_procs * static_cast<int>( sizeof( value_t ) );
    }

    comm_info.alltoallw(
        alltoallw_send.raw_ptr(), alltoallw_sendcounts.data(), alltoallw_sdispls.data(), alltoallw_sendtypes.data(),
        alltoallw_recv.raw_ptr(), alltoallw_recvcounts.data(), alltoallw_rdispls.data(), alltoallw_recvtypes.data()
    );

    for ( int src = 0; src < num_procs; ++src )
    {
        for ( int row = 0; row < num_procs; ++row )
        {
            const value_t expected = alltoallw_value( src, row, myid );
            if ( alltoallw_recv( src, row ) != expected )
            {
                fail(
                    "MPI_Alltoallw failed for src=%d, row=%d on rank %d: got %d, expected %d", src, row, myid,
                    alltoallw_recv( src, row ), expected
                );
            }
        }
    }

    scfd::communication::detail::type_free( column_type );
    if ( column_type != MPI_DATATYPE_NULL )
    {
        fail( "MPI_Type_free did not reset the datatype handle to MPI_DATATYPE_NULL" );
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
