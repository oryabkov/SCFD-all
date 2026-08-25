#include <iostream>

#include "test_backend_config.h"
#include <scfd/backend/backend.h>
#include <scfd/utils/log_std.h>

#if defined( SCFD_BACKEND_ENABLE_MPI )
#    include <scfd/communication/mpi_wrap.h>
#endif

#include "test_backend_runtime_common.h"

namespace
{

#if defined( SCFD_BACKEND_ENABLE_MPI )
int check_mpi_device_initialization( const scfd::communication::mpi_comm_info &comm )
{
    using backend_t = scfd_backend_tests::expected_backend;

    scfd::utils::log_std log;
    const int            device_with_log    = backend_t::init_device( log, comm, 0, true );
    const int            device_without_log = backend_t::init_device( comm, 0, true );

    if ( backend_t::is_device_backend() && ( device_with_log < 0 || device_without_log < 0 ) )
    {
        std::cout << scfd_backend_tests::expected_backend_configuration_name()
                  << ": FAILED communicator-aware device initialization" << std::endl;
        return 1;
    }
    if ( !backend_t::is_device_backend() && ( device_with_log != 0 || device_without_log != 0 ) )
    {
        std::cout << scfd_backend_tests::expected_backend_configuration_name()
                  << ": FAILED communicator-aware host initialization" << std::endl;
        return 2;
    }
    if ( device_with_log != device_without_log )
    {
        std::cout << scfd_backend_tests::expected_backend_configuration_name()
                  << ": FAILED communicator-aware initialization overload consistency" << std::endl;
        return 3;
    }
    return 0;
}
#endif

}

int main( int argc, char *argv[] )
{
#if defined( SCFD_BACKEND_ENABLE_MPI )
    scfd::communication::mpi_wrap mpi( argc, argv );
    const int                     mpi_init_status = check_mpi_device_initialization( mpi.comm_world() );
    if ( mpi_init_status != 0 )
    {
        return mpi_init_status;
    }
#else
    (void)argc;
    (void)argv;
#endif

    return scfd_backend_tests::run_backend_runtime_tests<scfd_backend_tests::expected_backend>(
        scfd_backend_tests::expected_backend_configuration_name()
    );
}
