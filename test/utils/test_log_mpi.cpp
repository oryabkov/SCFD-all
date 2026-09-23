// Copyright © 2026 SCFD contributors
// SPDX-License-Identifier: GPL-2.0-only

#include <iostream>

#include <mpi.h>

#include <scfd/utils/log_mpi.h>

int main( int argc, char **argv )
{
    if ( MPI_Init( &argc, &argv ) != MPI_SUCCESS )
        return 1;
    int rank = -1, size = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    scfd::utils::log_mpi log;
    const int            failed = log.comm_rank() != rank || log.comm_size() != size || size != 2;
    log.info_all_f( "rank %d of %d", log.comm_rank(), log.comm_size() );
    int       any_failed       = 0;
    const int reduction_status = MPI_Allreduce( &failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    if ( !any_failed && reduction_status == MPI_SUCCESS )
        log.info( "MPI logger check passed" );
    const int finalize_status = MPI_Finalize();
    return any_failed || reduction_status != MPI_SUCCESS || finalize_status != MPI_SUCCESS;
}
