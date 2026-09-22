// Copyright (C) 2026 SCFD contributors

#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/platform/platform.h>
#include <scfd/utils/log_std.h>

#ifdef PLATFORM_MPI
#    include <scfd/communication/mpi_comm.h>
#endif

#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) || defined( PLATFORM_SYCL ) )
#    include <scfd/utils/log_cformatted.h>
#    include <scfd/utils/log_msg_type.h>
#endif

#include "../backend/test_backend_config.h"
#include "../backend/test_backend_runtime_common.h"

namespace
{

#if !defined( PLATFORM_MPI )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::trivial<Ordinal, BigOrdinal>;
#elif defined( PLATFORM_SERIAL_CPU )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::serial_cpu_mpi<Ordinal, BigOrdinal>;
#elif defined( PLATFORM_OMP )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::omp_mpi<Ordinal, BigOrdinal>;
#elif defined( PLATFORM_CUDA )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::cuda_mpi<Ordinal, BigOrdinal>;
#elif defined( PLATFORM_HIP )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::hip_mpi<Ordinal, BigOrdinal>;
#elif defined( PLATFORM_SYCL )
template <class Ordinal, class BigOrdinal>
using expected_platform = scfd::platform::sycl_mpi<Ordinal, BigOrdinal>;
#endif

using default_platform     = scfd::platform::current;
using default_communicator = default_platform::communicator_type;

static_assert(
    std::is_same<default_platform, expected_platform<PLATFORM_ORDINAL, PLATFORM_BIG_ORDINAL>>::value,
    "default platform selection"
);
static_assert(
    std::is_same<default_platform::backend_type, scfd::backend::current>::value, "default backend selection"
);
static_assert( std::is_same<default_platform::ordinal_type, PLATFORM_ORDINAL>::value, "default local ordinal" );
static_assert(
    std::is_same<default_platform::big_ordinal_type, PLATFORM_BIG_ORDINAL>::value, "default global ordinal"
);
#ifdef PLATFORM_MPI
static_assert( std::is_same<default_communicator, scfd::communication::mpi_comm_info>::value, "MPI communicator" );
static_assert(
    std::is_same<default_platform::communication_environment_type, scfd::communication::mpi_wrap>::value,
    "MPI environment"
);
#else
static_assert(
    std::is_same<default_communicator, scfd::communication::trivial_comm<scfd::memory::host>>::value,
    "host-staged local communicator"
);
static_assert(
    std::is_same<
        default_platform::communication_environment_type,
        scfd::communication::trivial_platform<scfd::memory::host>>::value,
    "local queue owner"
);
#endif

int active_device()
{
#if defined( PLATFORM_CUDA )
    int device = -1;
    SCFD_CUDA_SAFE_CALL( cudaGetDevice( &device ) );
    return device;
#elif defined( PLATFORM_HIP )
    int device = -1;
    SCFD_HIP_SAFE_CALL( hipGetDevice( &device ) );
    return device;
#elif defined( PLATFORM_SYCL )
    const auto devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
    for ( std::size_t i = 0; i < devices.size(); ++i )
        if ( devices[i] == sycl_device_queue.get_device() )
            return static_cast<int>( i );
    return -1;
#else
    return 0;
#endif
}

#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) || defined( PLATFORM_SYCL ) )
class recording_log_basic
{
public:
    using log_msg_type = scfd::utils::log_msg_type;

    void msg( const std::string &text, log_msg_type type = log_msg_type::INFO, int = 1 )
    {
        messages_.push_back( { type, text } );
    }
    void set_verbosity( int = 1 )
    {
    }
    int count( log_msg_type type, const std::string &text ) const
    {
        int result = 0;
        for ( const auto &message : messages_ )
            if ( message.type == type && message.text.find( text ) != std::string::npos )
                ++result;
        return result;
    }

private:
    struct message
    {
        log_msg_type type;
        std::string  text;
    };
    std::vector<message> messages_;
};

using recording_log = scfd::utils::log_cformatted<recording_log_basic>;

int visible_device_count()
{
    int count = 0;
#    if defined( PLATFORM_CUDA )
    SCFD_CUDA_SAFE_CALL( cudaGetDeviceCount( &count ) );
#    elif defined( PLATFORM_HIP )
    SCFD_HIP_SAFE_CALL( hipGetDeviceCount( &count ) );
#    else
    count = static_cast<int>( ::sycl::device::get_devices( ::sycl::info::device_type::gpu ).size() );
#    endif
    return count;
}

template <class Comm>
bool check_initialization_log( const recording_log &log, const Comm &comm )
{
    using message_type                  = scfd::utils::log_msg_type;
    auto              node              = comm.split_type( MPI_COMM_TYPE_SHARED );
    const int         expected_warnings = node.myid() == 0 && node.num_procs() > visible_device_count() ? 1 : 0;
    const std::string rank_fields =
        "global_size = " + std::to_string( comm.num_procs ) + ", global_id = " + std::to_string( comm.myid ) + ",";
    // Capture every rank's calls without relying on interleaved MPI stdout.
    bool valid = log.count( message_type::INFO_ALL, "node_device_id = " ) == 1 &&
                 log.count( message_type::INFO_ALL, rank_fields ) == 1 &&
                 log.count( message_type::WARNING, "is wrapping " ) == expected_warnings;
#    if defined( PLATFORM_CUDA )
    valid = valid && log.count( message_type::INFO, "init_cuda:" ) > 0;
#    elif defined( PLATFORM_HIP )
    valid = valid && log.count( message_type::INFO, "init_hip:" ) > 0;
#    endif
    return valid;
}
#endif

template <class Comm>
int expected_device( const Comm &comm, int shift )
{
#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) || defined( PLATFORM_SYCL ) )
    auto      node  = comm.split_type( MPI_COMM_TYPE_SHARED );
    const int count = visible_device_count();
    return count > 0 ? ( node.myid() + shift ) % count : -1;
#else
    (void)comm;
    (void)shift;
    return 0;
#endif
}

template <class Platform>
int run_platform_tests( const typename Platform::communicator_type &comm, const char *case_name )
{
    using backend_t     = typename Platform::backend_type;
    using ordinal_t     = typename Platform::ordinal_type;
    using big_ordinal_t = typename Platform::big_ordinal_type;
    using comm_t        = typename Platform::communicator_type;
    static_assert( std::is_same<Platform, expected_platform<ordinal_t, big_ordinal_t>>::value, "platform selection" );
    static_assert(
        std::is_same<backend_t, scfd_backend_tests::expected_backend<ordinal_t>>::value, "local backend selection"
    );
    static_assert(
        std::is_same<typename backend_t::ordinal_type, ordinal_t>::value, "explicit platform ordinal reaches backend"
    );
    static_assert( std::is_same<comm_t, default_communicator>::value, "fixed platform communicator type" );
    static_assert( !std::is_base_of<backend_t, Platform>::value, "platform does not inherit its backend" );

    scfd::utils::log_std log;
    const std::string    name = std::string( backend_t::name() ) + "/platform/" + case_name;
#ifdef PLATFORM_MPI
    const int shift_cases = 2;
#else
    const int shift_cases = 1;
    if ( comm.num_procs != 1 || comm.myid != 0 || comm.queue == nullptr )
        return 1;
#endif
    for ( int shift = 0; shift < shift_cases; ++shift )
    {
#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) || defined( PLATFORM_SYCL ) )
        recording_log initialization_log;
        initialization_log.set_verbosity( 1 );
        const int with_log = Platform::init( initialization_log, comm, shift, true );
        if ( comm.all_reduce_sum( check_initialization_log( initialization_log, comm ) ? 0 : 1 ) != 0 )
        {
            log.error_f( "%s: MPI initialization did not forward the expected log diagnostics", name.c_str() );
            return 4;
        }
#else
        const int with_log = Platform::init( log, comm, shift, true );
#endif
        const int without_log = Platform::init( comm, shift, true );
        const int expected    = expected_device( comm, shift );
        int       status      = with_log != expected || without_log != expected || active_device() != expected;
        if ( comm.all_reduce_sum( status ) != 0 )
        {
            log.error_f( "%s: platform initialization selected an unexpected device", name.c_str() );
            return 2;
        }

        // Do not run local init_device(0) checks after communicator-based initialization.
        status = scfd_backend_tests::run_backend_runtime_tests<backend_t>( name.c_str(), false );
        if ( active_device() != expected )
            status = 3;
        if ( comm.all_reduce_sum( status != 0 ? 1 : 0 ) != 0 )
            return 3;
    }
    return 0;
}

#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) )
int run_direct_initialization_tests( const default_communicator &comm )
{
    for ( int shift = 0; shift < 2; ++shift )
    {
        recording_log log;
#    if defined( PLATFORM_CUDA )
        const int with_log    = scfd::utils::init_cuda_mpi( log, comm, shift, true );
        const int without_log = scfd::utils::init_cuda_mpi( comm, shift, true );
#    else
        const int with_log    = scfd::utils::init_hip_mpi( log, comm, shift, true );
        const int without_log = scfd::utils::init_hip_mpi( comm, shift, true );
#    endif
        const int  expected  = expected_device( comm, shift );
        const bool valid_log = check_initialization_log( log, comm );
        const int failed = with_log != expected || without_log != expected || active_device() != expected || !valid_log;
        if ( comm.all_reduce_sum( failed ) != 0 )
        {
            std::cerr << "direct MPI device initialization returned an unexpected device or log diagnostics\n";
            return 4;
        }
    }
    return 0;
}
#endif

int run_ordinal_cases( const default_communicator &comm )
{
    int status = run_platform_tests<default_platform>( comm, "default" );
    if ( status == 0 )
        status = run_platform_tests<expected_platform<int, std::ptrdiff_t>>( comm, "int/ptrdiff_t" );
    if ( status == 0 )
        status = run_platform_tests<expected_platform<std::ptrdiff_t, long long>>( comm, "ptrdiff_t/long-long" );
    return status;
}

}

int main( int argc, char *argv[] )
{
    default_platform::communication_environment_type environment( argc, argv );
    const auto                                       comm = environment.comm_world();
    try
    {
        int status = run_ordinal_cases( comm );
#ifdef PLATFORM_MPI
        if ( status == 0 )
        {
            // Reverse actual ranks even when running with only two processes.
            auto reordered = comm.split( 0, -comm.myid );
            status         = run_platform_tests<default_platform>( reordered.info(), "reordered-communicator" );
        }
        if ( status == 0 )
        {
            // Use subsets and reversed rank ordering, not an implicit MPI_COMM_WORLD.
            auto subset = comm.split( comm.myid % 2, -comm.myid );
            status      = run_platform_tests<default_platform>( subset.info(), "subcommunicator" );
#    if defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP )
            if ( status == 0 )
                status = run_direct_initialization_tests( subset.info() );
#    endif
        }
#endif
        return comm.all_reduce_sum( status != 0 ? 1 : 0 ) != 0 ? 1 : 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "platform runtime: " << error.what() << std::endl;
#ifdef PLATFORM_MPI
        MPI_Abort( comm.comm, 1 );
#endif
        return 1;
    }
}
