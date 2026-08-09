// Copyright (C) 2026 SCFD contributors

#if defined( PLATFORM_CUDA )
static const char *backend_name = "cuda";
#elif defined( PLATFORM_HIP )
static const char *backend_name = "hip";
#elif defined( PLATFORM_SYCL )
static const char *backend_name = "sycl";
#else
#    error "Define PLATFORM_CUDA, PLATFORM_HIP, or PLATFORM_SYCL for this test"
#endif

#ifndef SCFD_BACKEND_ENABLE_MPI
#    error "Define SCFD_BACKEND_ENABLE_MPI for this test"
#endif

#include <scfd/backend/backend.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>
#include "../backend/test_backend_runtime_common.h"

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm = mpi.comm_world();
    scfd::utils::log_mpi          log;

    const int device = scfd::backend::current::init_device( log, comm, 0, true );
    if ( device < 0 )
    {
        log.error_f( "%s backend init_device(log, comm, 0, true) returned %i", backend_name, device );
        return 1;
    }

    const int device_without_log = scfd::backend::current::init_device( comm, 0, true );
    if ( device_without_log < 0 )
    {
        log.error_f( "%s backend init_device(comm, 0, true) returned %i", backend_name, device_without_log );
        return 2;
    }
    if ( device_without_log != device )
    {
        log.error_f(
            "%s backend MPI init_device overloads returned different device ids: %i and %i",
            backend_name, device, device_without_log
        );
        return 3;
    }

    scfd::backend::current::synchronize();
    const int backend_runtime_status =
        scfd_backend_tests::run_backend_runtime_tests<scfd::backend::current>( backend_name );
    if ( backend_runtime_status != 0 )
        return 10 + backend_runtime_status;

    log.info_f( "PASSED" );
    return 0;
}
