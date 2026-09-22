// Copyright (C) 2026 SCFD contributors
// SPDX-License-Identifier: GPL-2.0-only

// Standalone regression: compile with mpic++ -std=c++11 -Iinclude and run
// with mpiexec -n 2 (also supports larger process counts). No CI helpers needed.
// Define PLATFORM_CUDA or PLATFORM_HIP with the corresponding compiler to
// additionally check MPI device initialization without the full backend SDK.

#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <mpi.h>

#include <scfd/utils/log_mpi.h>

#if defined( PLATFORM_CUDA )
#    include <scfd/utils/init_cuda_mpi.h>
#elif defined( PLATFORM_HIP )
#    include <scfd/utils/init_hip_mpi.h>
#endif

namespace
{

class stdout_capture
{
public:
    stdout_capture() : file_( std::tmpfile() ), saved_fd_( -1 )
    {
        if ( file_ == nullptr )
            throw std::runtime_error( "could not create stdout capture" );
        std::fflush( stdout );
        saved_fd_ = dup( fileno( stdout ) );
        if ( saved_fd_ < 0 || dup2( fileno( file_ ), fileno( stdout ) ) < 0 )
        {
            if ( saved_fd_ >= 0 )
                close( saved_fd_ );
            std::fclose( file_ );
            throw std::runtime_error( "could not redirect stdout" );
        }
    }

    ~stdout_capture()
    {
        if ( saved_fd_ >= 0 )
        {
            std::fflush( stdout );
            dup2( saved_fd_, fileno( stdout ) );
            close( saved_fd_ );
        }
        std::fclose( file_ );
    }

    stdout_capture( const stdout_capture & )            = delete;
    stdout_capture &operator=( const stdout_capture & ) = delete;

    std::string finish()
    {
        if ( std::fflush( stdout ) != 0 || dup2( saved_fd_, fileno( stdout ) ) < 0 )
            throw std::runtime_error( "could not restore stdout" );
        close( saved_fd_ );
        saved_fd_ = -1;
        std::rewind( file_ );
        std::string result;
        char        buffer[512];
        std::size_t count = 0;
        while ( ( count = std::fread( buffer, 1, sizeof( buffer ), file_ ) ) != 0 )
            result.append( buffer, count );
        if ( std::ferror( file_ ) )
            throw std::runtime_error( "could not read captured stdout" );
        return result;
    }

private:
    std::FILE *file_;
    int        saved_fd_;
};

void require( bool condition, const std::string &message )
{
    if ( !condition )
        throw std::runtime_error( message );
}

template <class Log>
void check_messages( Log &log )
{
    using message_type = typename Log::log_msg_type;
    stdout_capture capture;
    log.msg( "root", message_type::INFO );
    log.msg( "all", message_type::INFO_ALL );
    log.msg( "warning", message_type::WARNING );
    log.msg( "error", message_type::ERROR );
    log.set_verbosity( 0 );
    log.msg( "hidden-info", message_type::INFO );
    log.msg( "hidden-all", message_type::INFO_ALL );
    log.msg( "hidden-warning", message_type::WARNING );
    log.msg( "unfiltered-error", message_type::ERROR );
    log.set_verbosity( 2 );
    log.msg( "verbose", message_type::INFO_ALL, 2 );
    log.set_verbosity( 1 );
    log.msg( "hidden-verbose", message_type::INFO_ALL, 2 );

    const int rank = log.comm_rank();
    char      expected[512];
    std::snprintf(
        expected, sizeof( expected ),
        "%sINFO_ALL(%3d):all\nWARNING(%3d): warning\nERROR(%3d):   error\n"
        "ERROR(%3d):   unfiltered-error\nINFO_ALL(%3d):verbose\n",
        rank == 0 ? "INFO:         root\n" : "", rank, rank, rank, rank, rank
    );
    require( capture.finish() == expected, "incorrect rank filtering, prefixes, or verbosity" );
}

template <class Log>
void check_logger()
{
    static_assert( std::is_default_constructible<Log>::value, "default construction must remain available" );
    static_assert( std::is_constructible<Log, MPI_Comm>::value, "custom communicator construction" );
    static_assert( !std::is_convertible<MPI_Comm, Log>::value, "communicator construction must be explicit" );

    int world_rank = -1, world_size = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    Log default_log;
    Log brace_log = {};
    require(
        default_log.comm_rank() == world_rank && default_log.comm_size() == world_size,
        "default logger must use MPI_COMM_WORLD"
    );
    require(
        brace_log.comm_rank() == world_rank && brace_log.comm_size() == world_size,
        "brace initialization must retain the default communicator"
    );
    check_messages( default_log );

    bool null_rejected = false;
    try
    {
        Log invalid_log( MPI_COMM_NULL );
        (void)invalid_log;
    }
    catch ( const std::invalid_argument & )
    {
        null_rejected = true;
    }
    require( null_rejected, "MPI_COMM_NULL must be rejected before querying MPI" );

    MPI_Comm reversed = MPI_COMM_NULL, subset = MPI_COMM_NULL;
    require( MPI_Comm_split( MPI_COMM_WORLD, 0, -world_rank, &reversed ) == MPI_SUCCESS, "reverse split failed" );
    require(
        MPI_Comm_split( MPI_COMM_WORLD, world_rank % 2, -world_rank, &subset ) == MPI_SUCCESS, "subset split failed"
    );
    const MPI_Comm communicators[] = { MPI_COMM_WORLD, MPI_COMM_SELF, reversed, subset };
    for ( MPI_Comm comm : communicators )
    {
        int rank = -1, size = 0;
        MPI_Comm_rank( comm, &rank );
        MPI_Comm_size( comm, &size );
        Log log( comm );
        require( log.comm_rank() == rank && log.comm_size() == size, "logger must use its supplied communicator" );
        check_messages( log );
    }

    // Neither destruction of the earlier loggers nor freeing a communicator
    // invalidates the rank/size already cached by another logger.
    Log snapshot( reversed );
    require( MPI_Comm_free( &reversed ) == MPI_SUCCESS, "logger must not own its communicator" );
    require(
        snapshot.comm_rank() == world_size - 1 - world_rank && snapshot.comm_size() == world_size,
        "cached communicator metadata changed"
    );
    check_messages( snapshot );
    require( MPI_Comm_free( &subset ) == MPI_SUCCESS, "subset free failed" );
}

void check_formatted_messages()
{
    scfd::utils::log_mpi log( MPI_COMM_SELF );
    stdout_capture       capture;
    log.info_f( "root %d", 7 );
    log.info_all_f( "all %d", 8 );
    log.warning_f( "warning %d", 9 );
    log.error_f( "error %d", 10 );
    require(
        capture.finish() ==
            "INFO:         root 7\nINFO_ALL(  0):all 8\nWARNING(  0): warning 9\nERROR(  0):   error 10\n",
        "formatted logger must inherit the communicator constructor"
    );
}

#if defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP )
void check_device_initialization( MPI_Comm native_comm )
{
    int rank = -1, size = 0;
    MPI_Comm_rank( native_comm, &rank );
    MPI_Comm_size( native_comm, &size );
    scfd::communication::mpi_comm_info comm{ native_comm, size, rank, MPI_THREAD_SINGLE };
    int                                devices = 0;
#    if defined( PLATFORM_CUDA )
    SCFD_CUDA_SAFE_CALL( cudaGetDeviceCount( &devices ) );
    const char *local_prefix = "init_cuda:";
#    else
    SCFD_HIP_SAFE_CALL( hipGetDeviceCount( &devices ) );
    const char *local_prefix = "init_hip:";
#    endif
    require( devices > 0, "no GPU available for the requested initialization test" );
    auto       node             = comm.split_type( MPI_COMM_TYPE_SHARED );
    const int  expected_device  = ( node.myid() + 1 ) % devices;
    const bool expected_warning = node.myid() == 0 && node.num_procs() > devices;
    node.free();

    for ( int mode = 0; mode < 3; ++mode )
    {
        // Default, caller-supplied, and silent caller-supplied logging.
        scfd::utils::log_mpi log( native_comm );
        if ( mode == 2 )
            log.set_verbosity( 0 );
        stdout_capture capture;
#    if defined( PLATFORM_CUDA )
        const int device =
            mode == 0 ? scfd::utils::init_cuda_mpi( comm, 1, true ) : scfd::utils::init_cuda_mpi( log, comm, 1, true );
        int active_device = -1;
        SCFD_CUDA_SAFE_CALL( cudaGetDevice( &active_device ) );
#    else
        const int device =
            mode == 0 ? scfd::utils::init_hip_mpi( comm, 1, true ) : scfd::utils::init_hip_mpi( log, comm, 1, true );
        int active_device = -1;
        SCFD_HIP_SAFE_CALL( hipGetDevice( &active_device ) );
#    endif
        const std::string output = capture.finish();
        require( device == expected_device && active_device == expected_device, "device selection changed" );
        if ( mode == 2 )
        {
            require( output.empty(), "device initialization bypassed the supplied silent logger" );
            continue;
        }
        char mapping_prefix[32];
        std::snprintf( mapping_prefix, sizeof( mapping_prefix ), "INFO_ALL(%3d):", rank );
        require( output.find( mapping_prefix ) != std::string::npos, "missing communicator-local mapping prefix" );
        require(
            output.find( "global_id = " + std::to_string( rank ) + "," ) != std::string::npos,
            "mapping used a rank from a different communicator"
        );
        require(
            ( output.find( local_prefix ) != std::string::npos ) == ( rank == 0 ),
            "nested device initialization must honor communicator-root logging"
        );
        require(
            ( output.find( "WARNING(" ) != std::string::npos ) == expected_warning,
            "incorrect node-leader wrapping warning"
        );
    }
}

void check_device_communicators()
{
    int rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm reversed = MPI_COMM_NULL, subset = MPI_COMM_NULL;
    require( MPI_Comm_split( MPI_COMM_WORLD, 0, -rank, &reversed ) == MPI_SUCCESS, "device reverse split failed" );
    require( MPI_Comm_split( MPI_COMM_WORLD, rank % 2, -rank, &subset ) == MPI_SUCCESS, "device subset split failed" );
    check_device_initialization( MPI_COMM_WORLD );
    check_device_initialization( reversed );
    check_device_initialization( subset );
    require( MPI_Comm_free( &reversed ) == MPI_SUCCESS, "device reversed communicator free failed" );
    require( MPI_Comm_free( &subset ) == MPI_SUCCESS, "device subset communicator free failed" );
}
#endif

}

int main( int argc, char **argv )
{
    if ( MPI_Init( &argc, &argv ) != MPI_SUCCESS )
        return 1;
    int rank = -1, size = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    try
    {
        require( size >= 2, "run this test with at least two MPI processes" );
        check_logger<scfd::utils::log_mpi_basic>();
        check_logger<scfd::utils::log_mpi>();
        check_formatted_messages();
#if defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP )
        check_device_communicators();
#endif
        require( MPI_Barrier( MPI_COMM_WORLD ) == MPI_SUCCESS, "final barrier failed" );
        if ( rank == 0 )
            std::cout << "MPI logger communicator tests passed (" << size << " ranks)" << std::endl;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "MPI logger test, world rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
        return 1;
    }
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
