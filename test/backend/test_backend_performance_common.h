#ifndef __SCFD_TEST_BACKEND_PERFORMANCE_COMMON_H__
#define __SCFD_TEST_BACKEND_PERFORMANCE_COMMON_H__

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/functional/basic_ops.h>
#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/system_timer_event.h>
#ifdef PLATFORM_CUDA
#    include <scfd/utils/cuda_timer_event.h>
#endif
#ifdef PLATFORM_HIP
#    include <scfd/utils/hip_timer_event.h>
#endif

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

template <class Array>
struct fill_reverse
{
    fill_reverse( const Array &data_, std::size_t size_ ) : data( data_ ), size( size_ )
    {
    }

    Array       data;
    std::size_t size;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        data( idx ) = size - idx;
    }
};

template <class Array>
struct fill_grouped
{
    fill_grouped( const Array &data_, std::size_t group_size_ ) : data( data_ ), group_size( group_size_ )
    {
    }

    Array       data;
    std::size_t group_size;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        data( idx ) = idx / group_size;
    }
};

template <class Array>
struct fill_reverse_key_value
{
    fill_reverse_key_value( const Array &keys_, const Array &values_, std::size_t size_ )
        : keys( keys_ ), values( values_ ), size( size_ )
    {
    }

    Array       keys;
    Array       values;
    std::size_t size;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        const std::size_t key = size - idx;
        keys( idx )           = key;
        values( idx )         = key * 2;
    }
};

template <class Array>
struct fill_reduce_by_key_input
{
    fill_reduce_by_key_input( const Array &keys_, const Array &values_, std::size_t group_size_ )
        : keys( keys_ ), values( values_ ), group_size( group_size_ )
    {
    }

    Array       keys;
    Array       values;
    std::size_t group_size;

    __DEVICE_TAG__ void operator()( const std::size_t &idx ) const
    {
        keys( idx )   = idx / group_size;
        values( idx ) = 1;
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

template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_reverse<Array>> : std::true_type
{
};

template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_grouped<Array>> : std::true_type
{
};

template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_reverse_key_value<Array>> : std::true_type
{
};

template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_reduce_by_key_input<Array>> : std::true_type
{
};
#endif

namespace scfd_backend_tests
{

struct performance_result
{
    double for_each_ms;
    double copy_ms;
    double exclusive_scan_ms;
    double inclusive_scan_ms;
    double reduce_ms;
    double reduce_max_ms;
    double sequence_ms;
    double sort_ms;
    double unique_ms;
    double sort_by_key_ms;
    double reduce_by_key_ms;
    double set_intersection_ms;
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

template <class Backend>
struct performance_timer_event_type
{
    using type = scfd::utils::system_timer_event;
    static bool uses_device_events()
    {
        return false;
    }
};

#ifdef PLATFORM_CUDA
template <>
struct performance_timer_event_type<scfd::backend::cuda>
{
    using type = scfd::utils::cuda_timer_event;
    static bool uses_device_events()
    {
        return true;
    }
};
#endif

#ifdef PLATFORM_HIP
template <>
struct performance_timer_event_type<scfd::backend::hip>
{
    using type = scfd::utils::hip_timer_event;
    static bool uses_device_events()
    {
        return true;
    }
};
#endif

template <class Backend, class Operation, class Sync>
double measure_ms( Operation operation, Sync sync )
{
    using timer_t = typename performance_timer_event_type<Backend>::type;
    timer_t begin;
    timer_t end;
    begin.record();
    operation();
    if ( !performance_timer_event_type<Backend>::uses_device_events() )
    {
        sync();
    }
    end.record();
    return end.elapsed_time( begin );
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

inline int checked_backend_size( std::size_t size, const char *operation_name )
{
    if ( size > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
    {
        throw std::runtime_error(
            std::string( operation_name ) + " performance size exceeds the default backend operation ordinal range"
        );
    }
    return static_cast<int>( size );
}

inline std::size_t bounded_algorithm_size( std::size_t size )
{
    const std::size_t default_size = std::min( size, std::size_t( 1024 ) * 1024 );
    return std::min( size, env_size( "SCFD_BACKEND_PERF_ALGO_SIZE", default_size ) );
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

    ensure_size_is_supported<array_t>( size );

    array_t values;
    values.init( size );

    for_each_t for_each;
    for_each( fill_linear<array_t>( values ), size );
    for_each.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() { for_each( fill_linear<array_t>( values ), size ); }, [&]() { for_each.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
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
        const double elapsed = measure_ms<Backend>(
            [&]() { exclusive_scan( size, input.raw_ptr(), output.raw_ptr(), std::size_t( 5 ) ); },
            [&]() { exclusive_scan.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
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
double benchmark_copy( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t = typename Backend::template for_each_type<std::size_t>;
    using copy_t     = typename Backend::copy_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "copy" );

    array_t input;
    array_t output;
    input.init( size );
    output.init( size );

    for_each_t for_each;
    for_each( fill_linear<array_t>( input ), size );
    for_each.wait();

    copy_t backend_copy;
    backend_copy( op_size, input.raw_ptr(), output.raw_ptr() );
    backend_copy.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() { backend_copy( op_size, input.raw_ptr(), output.raw_ptr() ); },
            [&]() { backend_copy.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    typename array_t::view_type output_view( output, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = compute_linear_value( i );
        if ( output_view( i ) != expected )
        {
            output_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " copy verification failed at index " + std::to_string( i )
            );
        }
    }
    output_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_inclusive_scan( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t         = typename Backend::memory_type;
    using array_t          = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t       = typename Backend::template for_each_type<std::size_t>;
    using inclusive_scan_t = typename Backend::inclusive_scan_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "inclusive_scan" );

    array_t input;
    array_t output;
    input.init( size );
    output.init( size );

    for_each_t for_each;
    for_each( fill_constant<array_t>( input, 1 ), size );
    for_each.wait();

    inclusive_scan_t inclusive_scan;
    inclusive_scan( op_size, input.raw_ptr(), output.raw_ptr() );
    inclusive_scan.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() { inclusive_scan( op_size, input.raw_ptr(), output.raw_ptr() ); },
            [&]() { inclusive_scan.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    typename array_t::view_type output_view( output, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = i + 1;
        if ( output_view( i ) != expected )
        {
            output_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " inclusive_scan verification failed at index " + std::to_string( i )
            );
        }
    }
    output_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_reduce( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t = typename Backend::template for_each_type<std::size_t>;
    using reduce_t   = typename Backend::reduce_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "reduce" );

    array_t input;
    input.init( size );

    for_each_t for_each;
    for_each( fill_constant<array_t>( input, 1 ), size );
    for_each.wait();

    reduce_t    reduce;
    std::size_t sum = reduce( op_size, input.raw_ptr(), std::size_t( 0 ) );
    reduce.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() { sum = reduce( op_size, input.raw_ptr(), std::size_t( 0 ) ); }, [&]() { reduce.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    if ( sum != size )
    {
        throw std::runtime_error( std::string( backend_name ) + " reduce verification failed" );
    }

    return best_ms;
}

template <class Backend>
double benchmark_reduce_max( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using sequence_t = typename Backend::sequence_type;
    using reduce_t   = typename Backend::reduce_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "reduce max" );

    array_t input;
    input.init( size );

    sequence_t sequence;
    sequence( op_size, input.raw_ptr(), std::size_t( 1 ), std::size_t( 1 ) );
    sequence.wait();

    reduce_t    reduce;
    std::size_t max_value =
        reduce( op_size, input.raw_ptr(), std::size_t( 0 ), scfd::functional::maximum<std::size_t>() );
    reduce.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() {
                max_value =
                    reduce( op_size, input.raw_ptr(), std::size_t( 0 ), scfd::functional::maximum<std::size_t>() );
            },
            [&]() { reduce.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    if ( max_value != size )
    {
        throw std::runtime_error( std::string( backend_name ) + " reduce maximum verification failed" );
    }

    return best_ms;
}

template <class Backend>
double benchmark_sequence( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using sequence_t = typename Backend::sequence_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "sequence" );

    array_t output;
    output.init( size );

    sequence_t sequence;
    sequence( op_size, output.raw_ptr(), std::size_t( 2 ), std::size_t( 3 ) );
    sequence.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() { sequence( op_size, output.raw_ptr(), std::size_t( 2 ), std::size_t( 3 ) ); },
            [&]() { sequence.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    typename array_t::view_type output_view( output, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = 2 + 3 * i;
        if ( output_view( i ) != expected )
        {
            output_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " sequence verification failed at index " + std::to_string( i )
            );
        }
    }
    output_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_sort( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t = typename Backend::template for_each_type<std::size_t>;
    using sort_t     = typename Backend::sort_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "sort" );

    array_t values;
    values.init( size );

    for_each_t for_each;
    sort_t     sort;
    for_each( fill_reverse<array_t>( values, size ), size );
    for_each.wait();
    sort( op_size, values.raw_ptr() );
    sort.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        for_each( fill_reverse<array_t>( values, size ), size );
        for_each.wait();
        const double elapsed =
            measure_ms<Backend>( [&]() { sort( op_size, values.raw_ptr() ); }, [&]() { sort.wait(); } );
        best_ms = std::min( best_ms, elapsed );
    }

    typename array_t::view_type values_view( values, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i        = probe_indexes[probe_i];
        const std::size_t expected = i + 1;
        if ( values_view( i ) != expected )
        {
            values_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " sort verification failed at index " + std::to_string( i )
            );
        }
    }
    values_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_unique( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t   = typename Backend::memory_type;
    using array_t    = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t = typename Backend::template for_each_type<std::size_t>;
    using unique_t   = typename Backend::unique_type;

    const std::size_t group_size = 4;
    ensure_size_is_supported<array_t>( size );
    const int op_size       = checked_backend_size( size, "unique" );
    const int expected_size = checked_backend_size( ( size + group_size - 1 ) / group_size, "unique result" );

    array_t values;
    values.init( size );

    for_each_t for_each;
    unique_t   unique;
    int        unique_size = 0;

    for_each( fill_grouped<array_t>( values, group_size ), size );
    for_each.wait();
    unique_size = unique( op_size, values.raw_ptr() );
    unique.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        for_each( fill_grouped<array_t>( values, group_size ), size );
        for_each.wait();
        const double elapsed = measure_ms<Backend>(
            [&]() { unique_size = unique( op_size, values.raw_ptr() ); }, [&]() { unique.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    if ( unique_size != expected_size )
    {
        throw std::runtime_error( std::string( backend_name ) + " unique size verification failed" );
    }

    typename array_t::view_type values_view( values, true );
    const int                   probe_indexes[] = { 0, expected_size / 2, expected_size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const int         i        = probe_indexes[probe_i];
        const std::size_t expected = static_cast<std::size_t>( i );
        if ( values_view( i ) != expected )
        {
            values_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " unique verification failed at index " + std::to_string( i )
            );
        }
    }
    values_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_sort_by_key( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t      = typename Backend::memory_type;
    using array_t       = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t    = typename Backend::template for_each_type<std::size_t>;
    using sort_by_key_t = typename Backend::sort_by_key_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "sort_by_key" );

    array_t keys;
    array_t values;
    keys.init( size );
    values.init( size );

    for_each_t    for_each;
    sort_by_key_t sort_by_key;
    for_each( fill_reverse_key_value<array_t>( keys, values, size ), size );
    for_each.wait();
    sort_by_key( op_size, keys.raw_ptr(), values.raw_ptr() );
    sort_by_key.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        for_each( fill_reverse_key_value<array_t>( keys, values, size ), size );
        for_each.wait();
        const double elapsed = measure_ms<Backend>(
            [&]() { sort_by_key( op_size, keys.raw_ptr(), values.raw_ptr() ); }, [&]() { sort_by_key.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    typename array_t::view_type keys_view( keys, true );
    typename array_t::view_type values_view( values, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i              = probe_indexes[probe_i];
        const std::size_t expected_key   = i + 1;
        const std::size_t expected_value = expected_key * 2;
        if ( keys_view( i ) != expected_key || values_view( i ) != expected_value )
        {
            keys_view.release( false );
            values_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " sort_by_key verification failed at index " + std::to_string( i )
            );
        }
    }
    keys_view.release( false );
    values_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_reduce_by_key( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t        = typename Backend::memory_type;
    using array_t         = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using for_each_t      = typename Backend::template for_each_type<std::size_t>;
    using reduce_by_key_t = typename Backend::reduce_by_key_type;

    const std::size_t group_size = 8;
    ensure_size_is_supported<array_t>( size );
    const int op_size       = checked_backend_size( size, "reduce_by_key" );
    const int expected_size = checked_backend_size( ( size + group_size - 1 ) / group_size, "reduce_by_key result" );

    array_t keys_in;
    array_t values_in;
    array_t keys_out;
    array_t values_out;
    keys_in.init( size );
    values_in.init( size );
    keys_out.init( expected_size );
    values_out.init( expected_size );

    for_each_t      for_each;
    reduce_by_key_t reduce_by_key;
    for_each( fill_reduce_by_key_input<array_t>( keys_in, values_in, group_size ), size );
    for_each.wait();

    int output_size =
        reduce_by_key( op_size, keys_in.raw_ptr(), values_in.raw_ptr(), keys_out.raw_ptr(), values_out.raw_ptr() );
    reduce_by_key.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() {
                output_size = reduce_by_key(
                    op_size, keys_in.raw_ptr(), values_in.raw_ptr(), keys_out.raw_ptr(), values_out.raw_ptr()
                );
            },
            [&]() { reduce_by_key.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    if ( output_size != expected_size )
    {
        throw std::runtime_error( std::string( backend_name ) + " reduce_by_key size verification failed" );
    }

    typename array_t::view_type keys_view( keys_out, true );
    typename array_t::view_type values_view( values_out, true );
    const int                   probe_indexes[] = { 0, expected_size / 2, expected_size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const int         i              = probe_indexes[probe_i];
        const std::size_t expected_key   = static_cast<std::size_t>( i );
        const std::size_t first_index    = expected_key * group_size;
        const std::size_t expected_value = std::min( group_size, size - first_index );
        if ( keys_view( i ) != expected_key || values_view( i ) != expected_value )
        {
            keys_view.release( false );
            values_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " reduce_by_key verification failed at index " + std::to_string( i )
            );
        }
    }
    keys_view.release( false );
    values_view.release( false );

    return best_ms;
}

template <class Backend>
double benchmark_set_intersection( const char *backend_name, std::size_t size, int repeats )
{
    using memory_t           = typename Backend::memory_type;
    using array_t            = scfd::arrays::tensor0_array_nd<std::size_t, 1, memory_t>;
    using sequence_t         = typename Backend::sequence_type;
    using set_intersection_t = typename Backend::set_intersection_type;

    ensure_size_is_supported<array_t>( size );
    const int op_size = checked_backend_size( size, "set_intersection" );

    array_t set1;
    array_t set2;
    array_t result;
    set1.init( size );
    set2.init( size );
    result.init( size );

    sequence_t sequence;
    sequence( op_size, set1.raw_ptr(), std::size_t( 0 ), std::size_t( 1 ) );
    sequence( op_size, set2.raw_ptr(), std::size_t( 0 ), std::size_t( 1 ) );
    sequence.wait();

    set_intersection_t set_intersection;
    int output_size = set_intersection( op_size, set1.raw_ptr(), op_size, set2.raw_ptr(), result.raw_ptr() );
    set_intersection.wait();

    double best_ms = std::numeric_limits<double>::max();
    for ( int repeat = 0; repeat < repeats; ++repeat )
    {
        const double elapsed = measure_ms<Backend>(
            [&]() {
                output_size = set_intersection( op_size, set1.raw_ptr(), op_size, set2.raw_ptr(), result.raw_ptr() );
            },
            [&]() { set_intersection.wait(); }
        );
        best_ms = std::min( best_ms, elapsed );
    }

    if ( output_size != op_size )
    {
        throw std::runtime_error( std::string( backend_name ) + " set_intersection size verification failed" );
    }

    typename array_t::view_type result_view( result, true );
    const std::size_t           probe_indexes[] = { 0, size / 2, size - 1 };
    for ( int probe_i = 0; probe_i < 3; ++probe_i )
    {
        const std::size_t i = probe_indexes[probe_i];
        if ( result_view( i ) != i )
        {
            result_view.release( false );
            throw std::runtime_error(
                std::string( backend_name ) + " set_intersection verification failed at index " + std::to_string( i )
            );
        }
    }
    result_view.release( false );

    return best_ms;
}

template <class Backend>
performance_result
benchmark_backend( const char *backend_name, std::size_t size, std::size_t algorithm_size, int repeats )
{
    performance_result result;
    result.for_each_ms         = benchmark_for_each<Backend>( backend_name, size, repeats );
    result.copy_ms             = benchmark_copy<Backend>( backend_name, size, repeats );
    result.exclusive_scan_ms   = benchmark_exclusive_scan<Backend>( backend_name, size, repeats );
    result.inclusive_scan_ms   = benchmark_inclusive_scan<Backend>( backend_name, size, repeats );
    result.reduce_ms           = benchmark_reduce<Backend>( backend_name, size, repeats );
    result.reduce_max_ms       = benchmark_reduce_max<Backend>( backend_name, size, repeats );
    result.sequence_ms         = benchmark_sequence<Backend>( backend_name, size, repeats );
    result.sort_ms             = benchmark_sort<Backend>( backend_name, algorithm_size, repeats );
    result.unique_ms           = benchmark_unique<Backend>( backend_name, algorithm_size, repeats );
    result.sort_by_key_ms      = benchmark_sort_by_key<Backend>( backend_name, algorithm_size, repeats );
    result.reduce_by_key_ms    = benchmark_reduce_by_key<Backend>( backend_name, algorithm_size, repeats );
    result.set_intersection_ms = benchmark_set_intersection<Backend>( backend_name, algorithm_size, repeats );
    return result;
}

inline void print_result( const char *backend_name, const performance_result &result )
{
    std::cout << backend_name << ": for_each = " << result.for_each_ms << " ms, copy = " << result.copy_ms
              << " ms, exclusive_scan = " << result.exclusive_scan_ms
              << " ms, inclusive_scan = " << result.inclusive_scan_ms << " ms" << std::endl;
    std::cout << backend_name << ": reduce = " << result.reduce_ms << " ms, reduce_max = " << result.reduce_max_ms
              << " ms, sequence = " << result.sequence_ms << " ms" << std::endl;
    std::cout << backend_name << ": sort = " << result.sort_ms << " ms, unique = " << result.unique_ms
              << " ms, sort_by_key = " << result.sort_by_key_ms << " ms, reduce_by_key = " << result.reduce_by_key_ms
              << " ms, set_intersection = " << result.set_intersection_ms << " ms" << std::endl;
}

inline double speedup( double reference_ms, double backend_ms )
{
    return backend_ms > 0.0 ? reference_ms / backend_ms : std::numeric_limits<double>::infinity();
}

inline double total_ms( const performance_result &result )
{
    return result.for_each_ms + result.copy_ms + result.exclusive_scan_ms + result.inclusive_scan_ms +
           result.reduce_ms + result.reduce_max_ms + result.sequence_ms + result.sort_ms + result.unique_ms +
           result.sort_by_key_ms + result.reduce_by_key_ms + result.set_intersection_ms;
}

inline void print_openmp_comparison(
    const char *backend_name, const performance_result &openmp_result, const performance_result &backend_result
)
{
    std::cout << backend_name
              << ": speedup wrt openmp total = " << speedup( total_ms( openmp_result ), total_ms( backend_result ) )
              << ", for_each = " << speedup( openmp_result.for_each_ms, backend_result.for_each_ms )
              << ", copy = " << speedup( openmp_result.copy_ms, backend_result.copy_ms )
              << ", exclusive_scan = " << speedup( openmp_result.exclusive_scan_ms, backend_result.exclusive_scan_ms )
              << ", inclusive_scan = " << speedup( openmp_result.inclusive_scan_ms, backend_result.inclusive_scan_ms )
              << std::endl;
    std::cout << backend_name
              << ": speedup wrt openmp reduce = " << speedup( openmp_result.reduce_ms, backend_result.reduce_ms )
              << ", reduce_max = " << speedup( openmp_result.reduce_max_ms, backend_result.reduce_max_ms )
              << ", sequence = " << speedup( openmp_result.sequence_ms, backend_result.sequence_ms ) << std::endl;
    std::cout << backend_name
              << ": speedup wrt openmp sort = " << speedup( openmp_result.sort_ms, backend_result.sort_ms )
              << ", unique = " << speedup( openmp_result.unique_ms, backend_result.unique_ms )
              << ", sort_by_key = " << speedup( openmp_result.sort_by_key_ms, backend_result.sort_by_key_ms )
              << ", reduce_by_key = " << speedup( openmp_result.reduce_by_key_ms, backend_result.reduce_by_key_ms )
              << ", set_intersection = "
              << speedup( openmp_result.set_intersection_ms, backend_result.set_intersection_ms ) << std::endl;
}

template <class Backend>
int run_backend_performance_tests( const char *backend_name, bool require_acceleration )
{
    const std::size_t size           = env_size( "SCFD_BACKEND_PERF_SIZE", std::size_t( 8 ) * 1024 * 1024 * 10 );
    const std::size_t algorithm_size = bounded_algorithm_size( size );
    const int         repeats        = env_int( "SCFD_BACKEND_PERF_REPEATS", 5 );
    const double      min_speedup    = env_double( "SCFD_BACKEND_PERF_MIN_SPEEDUP", 1.05 );

    try
    {
        std::cout << backend_name << ": performance test size = " << size << ", algorithm size = " << algorithm_size
                  << ", repeats = " << repeats << std::endl;

        const performance_result backend_result =
            benchmark_backend<Backend>( backend_name, size, algorithm_size, repeats );
        print_result( backend_name, backend_result );

        if ( std::is_same<Backend, scfd::backend::omp>::value )
        {
            print_openmp_comparison( backend_name, backend_result, backend_result );
            std::cout << backend_name << ": openmp reference, acceleration check skipped" << std::endl;
            std::cout << backend_name << ": PASSED" << std::endl;
            return 0;
        }

        const performance_result openmp_result =
            benchmark_backend<scfd::backend::omp>( "openmp baseline", size, algorithm_size, repeats );
        print_result( "openmp baseline", openmp_result );
        print_openmp_comparison( backend_name, openmp_result, backend_result );

        if ( !require_acceleration )
        {
            std::cout << backend_name << ": PASSED" << std::endl;
            return 0;
        }

        const double total_speedup = speedup( total_ms( openmp_result ), total_ms( backend_result ) );
        if ( total_speedup < min_speedup )
        {
            std::cout << backend_name << ": FAILED acceleration check, openmp-relative total speedup " << total_speedup
                      << " < " << min_speedup << std::endl;
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
