#include <mpi.h>

#include <iostream>

int main( int argc, char **argv )
{
    if ( MPI_Init( &argc, &argv ) != MPI_SUCCESS )
        return 1;
    int rank  = 0;
    int size  = 0;
    int total = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    const int  status = MPI_Allreduce( &rank, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    const bool valid  = status == MPI_SUCCESS && size >= 2 && total == size * ( size - 1 ) / 2;
    if ( rank == 0 )
        std::cout << "MPI capability check " << ( valid ? "passed" : "failed" ) << " with " << size << " ranks\n";
    const int finalize_status = MPI_Finalize();
    return valid && finalize_status == MPI_SUCCESS ? 0 : 1;
}
