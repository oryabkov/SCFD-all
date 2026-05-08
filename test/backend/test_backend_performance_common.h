#ifndef __SCFD_TEST_BACKEND_PERFORMANCE_COMMON_H__
#define __SCFD_TEST_BACKEND_PERFORMANCE_COMMON_H__

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/device_tag.h>

namespace scfd_backend_tests
{

__DEVICE_TAG__ inline std::size_t compute_linear_value( std::size_t idx )
{
    unsigned long long value = static_cast<unsigned long long>( idx );
    for ( int i = 0; i < 128; ++i )
    {
        value = value * 2862933555777941757ULL + 3037000493ULL;
    }
    return static_cast<std::size_t>( value & 1023ULL ) + 1;
}

template <class Array>
struct fill_linear
{
    fill_linear( const Array &data_ ) : data( data_ )
    {
    }

    Array data;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        data( idx ) = compute_linear_value( idx );
    }
};

template <class Array>
struct fill_constant
{
    fill_constant( const Array &data_, std::size_t value_ ) : data( data_ ), value( value_ )
    {
    }

    Array       data;
    std::size_t value;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        data( idx ) = value;
    }
};

}

#ifdef PLATFORM_SYCL
template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_linear<Array>> : std::true_type
{
};

template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_constant<Array>> : std::true_type
{
};
#endif

namespace scfd_backend_tests
{

struct performance_result
{
    double for_each_ms;
    double exclusive_scan_ms;
};

inline int env_int( const char *name, int default_value )
{
    const char *value = std::getenv( name );
    if ( value == nullptr || *value == '\0' )
    {
        return default_value;
    }
    return std::max( 1, std::atoi( value ) );
}

inline std::size_t env_size( const char *name, std::size_t default_value )
{
    const char *value = std::getenv( name );
    if ( value == nullptr || *value == '\0' )
    {
        return default_value;
    }

    char *end                       = nullptr;
    errno                           = 0;
    const unsigned long long parsed = std::strtoull( value, &end, 10 );
    if ( errno == ERANGE || end == value || *end != '\0' )
    {
        throw std::runtime_error( std::string( "invalid " ) + name + " value: " + value );
    }
    if ( parsed == 0 )
    {
        return 1;
    }
    if ( parsed > static_cast<unsigned long long>( std::numeric_limits<std::size_t>::max() ) )
    {
        throw std::runtime_error( std::string( name ) + " does not fit into std::size_t" );
    }
    return static_cast<std::size_t>( parsed );
}

inline double env_double( const char *name, double default_value )
{
    const char *value = std::getenv( name );
    if ( value == nullptr || *value == '\0' )
    {
        return default_value;
    }
    return std::atof( value );
}

template <class Clock>
double elapsed_ms( const typename Clock::time_point &begin, const typename Clock::time_point &end )
{
    return std::chrono::duration<double, std::milli>( end - begin ).count();
}

template <class Array>
void ensure_size_is_supported( std::size_t size )
{
    using ordinal_t = typename Array::ordinal_type;
    if ( size > static_cast<std::size_t>( std::numeric_limits<ordinal_t>::max() ) )
    {
        throw std::runtime_error(
            "requested performance size exceeds SCFD array ordinal type range; rebuild with a wider "
            "SCFD_ARRAYS_ORDINAL_TYPE"
        );
    }
}

template <class Backend>
struct performance_exclusive_scan_type
{
    using type = typename Backend::exclusive_scan_type;
};

template <>
struct performance_exclusive_scan_type<scfd::backend::serial_cpu>
{
    using type = scfd::serial_cpu_exclusive_scan<std::size_t>;
};

#ifdef PLATFORM_OMP
template <>
struct performance_exclusive_scan_type<scfd::backend::omp>
{
    using type = scfd::omp_exclusive_scan<std::size_t>;
};
#endif

#ifdef PLATFORM_CUDA
template <>
struct performance_exclusive_scan_type<scfd::backend::cuda>
{
    using type = scfd::thrust_exclusive_scan<std::size_t>;
};
#endif

#ifdef PLATFORM_HIP
template <>
struct performance_exclusive_scan_type<scfd::backend::hip>
{
    using type = scfd::thrust_exclusive_scan<std::size_t>;
};
#endif

#ifdef PLATFORM_SYCL
template <>
struct performance_exclusive_scan_type<scfd::backend::sycl>
{
    using type = scfd::sycl_exclusive_scan<std::size_t>;
};
#endif

template <class Backend>
double benchmark_for_each( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t = typename Backend::template for_each_type<std::size_t>;
    using clock_t    = std::chrono::steady_clock;

    ensure_size_is_supported<array_t>( size );

    array_t values;
    values.init( size );

    for_each_t for_each;
    for_each( fill_linear<array_t>( values ), size );
    for_each.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const auto begin = clock_t::now();
        for_each( fill_linear<array_t>( values ), size );
        for_each.wait();
        const auto end = clock_t::now();
        best_ms        = std::min( best_ms, elapsed_ms<clock_t>( begin, end ) );
    }

    typename array_t::view_type values_view( values, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = compute_linear_value( i );
        if ( values_view( i ) != expected )
        {
            values_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " for_each verification failed at index " + std::to_string( i )
            );
        }
    }
    values_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_exclusive_scan( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t         = typename Backend::memory_type;
    using array_t          = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t       = typename Backend::template for_each_type<std::size_t>;
    using exclusive_scan_t = typename performance_exclusive_scan_type<Backend>::type;
    using clock_t          = std::chrono::steady_clock;

    ensure_size_is_supported<array_t>( size );

    array_t input;
    array_t output;
    input.init( size );
    output.init( size );

    for_each_t for_each;
    for_each( fill_constant<array_t>( input, 1 ), size );
    for_each.wait();

    exclusive_scan_t exclusive_scan;
    exclusive_scan( size, input.raw_ptr(), output.raw_ptr(), std::size_t( 5 ) );
    exclusive_scan.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const auto begin = clock_t::now();
        exclusive_scan( size, input.raw_ptr(), output.raw_ptr(), std::size_t( 5 ) );
        exclusive_scan.wait();
        const auto end = clock_t::now();
        best_ms        = std::min( best_ms, elapsed_ms<clock_t>( begin, end ) );
    }

    typename array_t::view_type output_view( output, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = i + 5;
        if ( output_view( i ) != expected )
        {
            output_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " exclusive_scan verification failed at index " + std::to_string( i )
            );
        }
    }
    output_view.release( false );

    return best_ms;
}

template <class Backend>
performance_result benchmark_backend( const char *backend_name, std::size_t size, int repeats )
{
    performance_result result;
    result.for_each_ms       = benchmark_for_each<Backend>( backend_name, size, repeats );
    result.exclusive_scan_ms = benchmark_exclusive_scan<Backend>( backend_name, size, repeats );
    return result;
}

inline void print_result( const char *backend_name, const performance_result &result )
{
    std::cout << backend_name << ": for_each = " << result.for_each_ms
              << " ms, exclusive_scan = " << result.exclusive_scan_ms << " ms" << std::endl;
}

template <class Backend>
int run_backend_performance_tests( const char *backend_name, bool require_acceleration )
{
    const std::size_t size        = env_size( "SCFD_BACKEND_PERF_SIZE", std::size_t( 8 ) * 1024 * 1024 * 10 );
    const int         repeats     = env_int( "SCFD_BACKEND_PERF_REPEATS", 5 );
    const double      min_speedup = env_double( "SCFD_BACKEND_PERF_MIN_SPEEDUP", 1.05 );

    try
    {
        std::cout << backend_name << ": performance test size = " << size << ", repeats = " << repeats << std::endl;

        const performance_result backend_result = benchmark_backend<Backend>( backend_name, size, repeats );
        print_result( backend_name, backend_result );

        if ( !require_acceleration )
        {
            std::cout << backend_name << ": PASSED" << std::endl;
            return 0;
        }

        const performance_result serial_result =
            benchmark_backend<scfd::backend::serial_cpu>( "serial_cpu baseline", size, repeats );
        print_result( "serial_cpu baseline", serial_result );

        const double backend_total    = backend_result.for_each_ms + backend_result.exclusive_scan_ms;
        const double serial_total     = serial_result.for_each_ms + serial_result.exclusive_scan_ms;
        const double total_speedup    = serial_total / backend_total;
        const double for_each_speedup = serial_result.for_each_ms / backend_result.for_each_ms;
        const double scan_speedup     = serial_result.exclusive_scan_ms / backend_result.exclusive_scan_ms;

        std::cout << backend_name << ": speedup total = " << total_speedup << ", for_each = " << for_each_speedup
                  << ", exclusive_scan = " << scan_speedup << std::endl;

        if ( total_speedup < min_speedup )
        {
            std::cout << backend_name << ": FAILED acceleration check, total speedup " << total_speedup << " < "
                      << min_speedup << std::endl;
            return 2;
        }
    }
    catch ( const std::exception &err )
    {
        std::cout << backend_name << ": FAILED with exception: " << err.what() << std::endl;
        return 1;
    }

    std::cout << backend_name << ": PASSED" << std::endl;
    return 0;
}

}

#endif
