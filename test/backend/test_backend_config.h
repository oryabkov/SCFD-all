#ifndef SCFD_TEST_BACKEND_CONFIG_H
#define SCFD_TEST_BACKEND_CONFIG_H

#if ( defined( PLATFORM_SERIAL_CPU ) + defined( PLATFORM_OMP ) + defined( PLATFORM_CUDA ) + defined( PLATFORM_HIP ) +  \
      defined( PLATFORM_SYCL ) ) != 1
#    error "Define exactly one SCFD backend platform"
#endif

#if defined( PLATFORM_SERIAL_CPU )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
#        include <scfd/backend/serial_cpu_mpi.h>
#    else
#        include <scfd/backend/serial_cpu.h>
#    endif
#elif defined( PLATFORM_OMP )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
#        include <scfd/backend/omp_mpi.h>
#    else
#        include <scfd/backend/omp.h>
#    endif
#elif defined( PLATFORM_CUDA )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
#        include <scfd/backend/cuda_mpi.h>
#    else
#        include <scfd/backend/cuda.h>
#    endif
#elif defined( PLATFORM_HIP )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
#        include <scfd/backend/hip_mpi.h>
#    else
#        include <scfd/backend/hip.h>
#    endif
#elif defined( PLATFORM_SYCL )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
#        include <scfd/backend/sycl_mpi.h>
#    else
#        include <scfd/backend/sycl.h>
#    endif
#endif

namespace scfd_backend_tests
{

#if defined( PLATFORM_SERIAL_CPU )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
using expected_backend = scfd::backend::serial_cpu_mpi;
#    else
using expected_backend = scfd::backend::serial_cpu;
#    endif
inline const char *expected_backend_name()
{
    return "serial_cpu";
}
constexpr bool expected_backend_requires_acceleration = false;
#elif defined( PLATFORM_OMP )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
using expected_backend = scfd::backend::omp_mpi;
#    else
using expected_backend = scfd::backend::omp;
#    endif
inline const char *expected_backend_name()
{
    return "omp";
}
constexpr bool expected_backend_requires_acceleration = false;
#elif defined( PLATFORM_CUDA )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
using expected_backend = scfd::backend::cuda_mpi;
#    else
using expected_backend = scfd::backend::cuda;
#    endif
inline const char *expected_backend_name()
{
    return "cuda";
}
constexpr bool expected_backend_requires_acceleration = true;
#elif defined( PLATFORM_HIP )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
using expected_backend = scfd::backend::hip_mpi;
#    else
using expected_backend = scfd::backend::hip;
#    endif
inline const char *expected_backend_name()
{
    return "hip";
}
constexpr bool expected_backend_requires_acceleration = true;
#elif defined( PLATFORM_SYCL )
#    if defined( SCFD_BACKEND_ENABLE_MPI )
using expected_backend = scfd::backend::sycl_mpi;
#    else
using expected_backend = scfd::backend::sycl;
#    endif
inline const char *expected_backend_name()
{
    return "sycl";
}
constexpr bool expected_backend_requires_acceleration = true;
#endif

inline const char *expected_backend_configuration_name()
{
#if defined( SCFD_BACKEND_ENABLE_MPI )
    static const char suffix[] = "_mpi";
    static char       name[32] = {};
    if ( name[0] == '\0' )
    {
        const char *base = expected_backend_name();
        char       *out  = name;
        while ( *base != '\0' )
        {
            *out++ = *base++;
        }
        const char *part = suffix;
        while ( *part != '\0' )
        {
            *out++ = *part++;
        }
        *out = '\0';
    }
    return name;
#else
    return expected_backend_name();
#endif
}

}

#endif
