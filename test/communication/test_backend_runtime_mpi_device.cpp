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

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm = mpi.comm_world();
    scfd::utils::log_mpi          log;

    const int device = scfd::backend::runtime::init_device_mpi<true>( log, comm );
    if ( device < 0 )
    {
        log.error_f( "%s backend init_device_mpi<true> returned %i", backend_name, device );
        return 1;
    }

    scfd::backend::runtime::synchronize();
    log.info_f( "PASSED" );
    return 0;
}
