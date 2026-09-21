// Copyright (C) 2026 SCFD contributors

#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <scfd/platform/platform.h>
#include <scfd/utils/log_std.h>

#ifdef PLATFORM_MPI
#    include <scfd/communication/mpi_comm.h>
#endif

#include "../backend/test_backend_runtime_common.h"

namespace
{

#if !defined( PLATFORM_MPI )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::local<Ordinal, BigOrdinal, Comm>;
#elif defined( PLATFORM_SERIAL_CPU )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::serial_cpu_mpi<Ordinal, BigOrdinal, Comm>;
#elif defined( PLATFORM_OMP )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::omp_mpi<Ordinal, BigOrdinal, Comm>;
#elif defined( PLATFORM_CUDA )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::cuda_mpi<Ordinal, BigOrdinal, Comm>;
#elif defined( PLATFORM_HIP )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::hip_mpi<Ordinal, BigOrdinal, Comm>;
#elif defined( PLATFORM_SYCL )
template <class Ordinal, class BigOrdinal, class Comm>
using expected_platform = scfd::platform::sycl_mpi<Ordinal, BigOrdinal, Comm>;
#endif

using default_platform     = scfd::platform::current<>;
using default_communicator = default_platform::communicator_type;

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

struct custom_communicator : default_communicator
{
    explicit custom_communicator( const default_communicator &comm ) : default_communicator( comm )
    {
    }
};

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

template <class Comm>
int expected_device( const Comm &comm, int shift )
{
#if defined( PLATFORM_MPI ) && ( defined( PLATFORM_CUDA ) || defined( PLATFORM_HIP ) || defined( PLATFORM_SYCL ) )
    auto node  = comm.split_type( MPI_COMM_TYPE_SHARED );
    int  count = 0;
#    if defined( PLATFORM_CUDA )
    SCFD_CUDA_SAFE_CALL( cudaGetDeviceCount( &count ) );
#    elif defined( PLATFORM_HIP )
    SCFD_HIP_SAFE_CALL( hipGetDeviceCount( &count ) );
#    else
    count = static_cast<int>( ::sycl::device::get_devices( ::sycl::info::device_type::gpu ).size() );
#    endif
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
    static_assert(
        std::is_same<Platform, expected_platform<ordinal_t, big_ordinal_t, comm_t>>::value, "platform selection"
    );
    static_assert( std::is_same<backend_t, scfd::backend::current<ordinal_t>>::value, "local backend selection" );
    static_assert( std::is_same<typename Platform::runtime_type, Platform>::value, "platform runtime" );
    static_assert( std::is_base_of<backend_t, Platform>::value, "platform exposes its backend" );
    static_assert(
        std::is_same<typename Platform::memory_type, typename backend_t::memory_type>::value, "memory alias"
    );
    static_assert(
        std::is_same<typename Platform::for_each_type, typename backend_t::for_each_type>::value,
        "local algorithm ordinal"
    );
    static_assert(
        std::is_same<
            typename Platform::template for_each_nd_type<3>, typename backend_t::template for_each_nd_type<3>>::value,
        "ND algorithm ordinal"
    );
    static_assert(
        std::is_same<typename Platform::reduce_type, typename backend_t::reduce_type>::value, "reduction ordinal"
    );

    scfd::utils::log_std log;
    const std::string    name = std::string( Platform::name() ) + "/platform/" + case_name;
#ifdef PLATFORM_MPI
    const int shift_cases = 2;
#else
    const int shift_cases = 1;
    if ( comm.num_procs != 1 || comm.myid != 0 || comm.queue == nullptr )
        return 1;
#endif
    for ( int shift = 0; shift < shift_cases; ++shift )
    {
        const int with_log    = Platform::init( log, comm, shift, true );
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

int run_ordinal_cases( const default_communicator &comm )
{
    int status = run_platform_tests<default_platform>( comm, "default" );
    if ( status == 0 )
        status = run_platform_tests<scfd::platform::current<int, std::ptrdiff_t>>( comm, "int/ptrdiff_t" );
    if ( status == 0 )
        status = run_platform_tests<scfd::platform::current<std::ptrdiff_t, long long>>( comm, "ptrdiff_t/long-long" );
    if ( status == 0 )
        status = run_platform_tests<scfd::platform::current<int, long long, custom_communicator>>(
            custom_communicator( comm ), "custom-communicator"
        );
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
            // Use subsets and reversed rank ordering, not an implicit MPI_COMM_WORLD.
            auto subset = comm.split( comm.myid % 2, -comm.myid );
            status      = run_platform_tests<default_platform>( subset.info(), "subcommunicator" );
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
