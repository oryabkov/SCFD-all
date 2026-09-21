#ifndef SCFD_TEST_BACKEND_CONFIG_H
#define SCFD_TEST_BACKEND_CONFIG_H

#include <cstddef>
#include <type_traits>

#if ( defined( PLATFORM_SERIAL_CPU ) + defined( PLATFORM_OMP ) + defined( PLATFORM_CUDA ) + defined( PLATFORM_HIP ) +  \
      defined( PLATFORM_SYCL ) ) != 1
#    error "Define exactly one SCFD backend platform"
#endif

#if defined( PLATFORM_SERIAL_CPU )
#    include <scfd/backend/serial_cpu.h>
#elif defined( PLATFORM_OMP )
#    include <scfd/backend/omp.h>
#elif defined( PLATFORM_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( PLATFORM_HIP )
#    include <scfd/backend/hip.h>
#elif defined( PLATFORM_SYCL )
#    include <scfd/backend/sycl.h>
#endif

namespace scfd_backend_tests
{

#if defined( PLATFORM_SERIAL_CPU )
template <class Ordinal = PLATFORM_ORDINAL>
using expected_backend = scfd::backend::serial_cpu<Ordinal>;
inline const char *expected_backend_name()
{
    return "serial_cpu";
}
constexpr bool expected_backend_requires_acceleration = false;
#elif defined( PLATFORM_OMP )
template <class Ordinal = PLATFORM_ORDINAL>
using expected_backend = scfd::backend::omp<Ordinal>;
inline const char *expected_backend_name()
{
    return "omp";
}
constexpr bool expected_backend_requires_acceleration = false;
#elif defined( PLATFORM_CUDA )
template <class Ordinal = PLATFORM_ORDINAL>
using expected_backend = scfd::backend::cuda<Ordinal>;
inline const char *expected_backend_name()
{
    return "cuda";
}
constexpr bool expected_backend_requires_acceleration = true;
#elif defined( PLATFORM_HIP )
template <class Ordinal = PLATFORM_ORDINAL>
using expected_backend = scfd::backend::hip<Ordinal>;
inline const char *expected_backend_name()
{
    return "hip";
}
constexpr bool expected_backend_requires_acceleration = true;
#elif defined( PLATFORM_SYCL )
template <class Ordinal = PLATFORM_ORDINAL>
using expected_backend = scfd::backend::sycl<Ordinal>;
inline const char *expected_backend_name()
{
    return "sycl";
}
constexpr bool expected_backend_requires_acceleration = true;
#endif

template <class Ordinal>
struct backend_ordinal
{
    using type = expected_backend<Ordinal>;
};

static_assert(
    std::is_same<typename expected_backend<>::ordinal_type, PLATFORM_ORDINAL>::value,
    "the default backend ordinal follows PLATFORM_ORDINAL"
);

template <class Test>
int run_backend_ordinal_tests( Test test )
{
    int status = test( backend_ordinal<PLATFORM_ORDINAL>(), "default" );
    if ( status == 0 && !std::is_same<PLATFORM_ORDINAL, int>::value )
    {
        status = test( backend_ordinal<int>(), "int" );
    }
    if ( status == 0 && !std::is_same<PLATFORM_ORDINAL, std::ptrdiff_t>::value )
    {
        status = test( backend_ordinal<std::ptrdiff_t>(), "ptrdiff_t" );
    }
    return status;
}

inline const char *expected_backend_configuration_name()
{
    return expected_backend_name();
}

}

#endif
