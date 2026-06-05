// Copyright (C) 2026 SCFD contributors

#define PLATFORM_SERIAL_CPU
#define SCFD_BACKEND_ENABLE_MPI

#include <scfd/backend/backend.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm = mpi.comm_world();
    scfd::utils::log_mpi          log;

    const int device_with_log = scfd::backend::current::init_device( log, comm, 0, true );
    if ( device_with_log != 0 )
    {
        log.error_f( "serial backend init_device(log, comm, 0, true) returned %i instead of 0", device_with_log );
        return 1;
    }

    const int device_without_log = scfd::backend::current::init_device( comm );
    if ( device_without_log != 0 )
    {
        log.error_f( "serial backend init_device(comm) returned %i instead of 0", device_without_log );
        return 2;
    }

    const int device = scfd::backend::current::init_device();
    if ( device != 0 )
    {
        log.error_f( "serial backend init_device returned %i instead of 0", device );
        return 3;
    }

    log.info_f( "PASSED" );
    return 0;
}
