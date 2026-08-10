#ifndef __SCFD_TEST_BACKEND_RUNTIME_COMMON_H__
#define __SCFD_TEST_BACKEND_RUNTIME_COMMON_H__

#include <exception>
#include <iostream>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/value_pair.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_std.h>

namespace scfd_backend_tests
{

template <class Array>
struct fill_value_pairs_by_index
{
    fill_value_pairs_by_index( const Array &data_ ) : data( data_ )
    {
    }

    Array data;

    __DEVICE_TAG__ void operator()( const int &idx ) const
    {
        data( idx ) = scfd::backend::make_value_pair( idx, idx * idx + 1 );
    }
};

}

#ifdef PLATFORM_SYCL
template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_value_pairs_by_index<Array>> : std::true_type
{
};
#endif

namespace scfd_backend_tests
{

inline bool valid_c_string( const char *value )
{
    return value != 0 && value[0] != '\0';
}

template <class Runtime>
int check_init_device_result( const char *backend_name, const char *call_name, int device )
{
    if ( Runtime::is_device_backend() && device < 0 )
    {
        std::cout << backend_name << ": FAILED " << call_name << " result" << std::endl;
        return 1;
    }
    if ( !Runtime::is_device_backend() && device != 0 )
    {
        std::cout << backend_name << ": FAILED host " << call_name << " result" << std::endl;
        return 2;
    }
    return 0;
}

template <class Runtime>
int check_device_memory_info(
    const char *backend_name, const char *call_name, const scfd::backend::device_memory_info &info
)
{
    if ( info.free_bytes_known != Runtime::reports_free_memory() )
    {
        std::cout << backend_name << ": FAILED " << call_name << " free-bytes capability flag" << std::endl;
        return 1;
    }
    if ( info.total_bytes_known != Runtime::reports_total_memory() )
    {
        std::cout << backend_name << ": FAILED " << call_name << " total-bytes capability flag" << std::endl;
        return 2;
    }
    if ( info.free_bytes_known && info.total_bytes_known && info.free_bytes > info.total_bytes )
    {
        std::cout << backend_name << ": FAILED " << call_name << " memory info ordering" << std::endl;
        return 3;
    }
    if ( Runtime::reports_total_memory() && info.total_bytes == 0 )
    {
        std::cout << backend_name << ": FAILED " << call_name << " total-bytes value" << std::endl;
        return 4;
    }
    if ( !info.free_bytes_known && info.free_bytes != 0 )
    {
        std::cout << backend_name << ": FAILED " << call_name << " unknown free-bytes value" << std::endl;
        return 5;
    }
    if ( !info.total_bytes_known && info.total_bytes != 0 )
    {
        std::cout << backend_name << ": FAILED " << call_name << " unknown total-bytes value" << std::endl;
        return 6;
    }
    return 0;
}

inline int check_host_memory_info( const char *backend_name, const scfd::backend::host_memory_info &host_info )
{
    if ( host_info.process_memory_known && host_info.rss_bytes == 0 )
    {
        std::cout << backend_name << ": FAILED host memory RSS value" << std::endl;
        return 1;
    }
    if ( host_info.peak_rss_known && host_info.process_memory_known && host_info.peak_rss_bytes < host_info.rss_bytes )
    {
        std::cout << backend_name << ": FAILED host memory peak RSS ordering" << std::endl;
        return 2;
    }
    if ( host_info.virtual_memory_known && host_info.process_memory_known &&
         host_info.virtual_bytes < host_info.rss_bytes )
    {
        std::cout << backend_name << ": FAILED host memory virtual/RSS ordering" << std::endl;
        return 3;
    }
    if ( host_info.system_memory_known && host_info.system_total_bytes == 0 )
    {
        std::cout << backend_name << ": FAILED host system total memory value" << std::endl;
        return 4;
    }
    if ( host_info.system_memory_known && host_info.system_available_bytes > host_info.system_total_bytes )
    {
        std::cout << backend_name << ": FAILED host memory available/total ordering" << std::endl;
        return 5;
    }
    if ( host_info.system_memory_known && host_info.system_free_bytes > host_info.system_total_bytes )
    {
        std::cout << backend_name << ": FAILED host memory free/total ordering" << std::endl;
        return 6;
    }
    return 0;
}

template <class Backend>
int run_backend_runtime_tests( const char *backend_name )
{
    using runtime_t       = typename Backend::runtime_type;
    using memory_t        = typename Backend::memory_type;
    using for_each_t      = typename Backend::template for_each_type<int>;
    using pair_t          = scfd::backend::value_pair<int, int>;
    using array_t         = scfd::arrays::tensor0_array_nd<pair_t, 1, memory_t>;
    using device_info_t   = typename Backend::device_memory_info_type;
    using host_info_t     = typename Backend::host_memory_info_type;
    using device_alias_t  = scfd::backend::device_memory_info;
    using host_alias_t    = scfd::backend::host_memory_info;
    using timer_alias_t   = scfd::backend::timer_event;
    using current_alias_t = scfd::backend::current;

    try
    {
        if ( !std::is_same<Backend, current_alias_t>::value )
        {
            std::cout << backend_name << ": FAILED current backend type check" << std::endl;
            return 10;
        }
        if ( !std::is_same<runtime_t, scfd::backend::runtime>::value )
        {
            std::cout << backend_name << ": FAILED runtime type check" << std::endl;
            return 11;
        }
        if ( !std::is_same<typename runtime_t::timer_event_type, timer_alias_t>::value )
        {
            std::cout << backend_name << ": FAILED timer_event type check" << std::endl;
            return 12;
        }
        if ( !std::is_same<device_info_t, device_alias_t>::value )
        {
            std::cout << backend_name << ": FAILED device memory info type check" << std::endl;
            return 26;
        }
        if ( !std::is_same<host_info_t, host_alias_t>::value )
        {
            std::cout << backend_name << ": FAILED host memory info type check" << std::endl;
            return 27;
        }
        if ( !valid_c_string( runtime_t::name() ) )
        {
            std::cout << backend_name << ": FAILED backend name" << std::endl;
            return 28;
        }
        if ( !runtime_t::is_device_backend() && runtime_t::uses_device_timer() )
        {
            std::cout << backend_name << ": FAILED device timer capability flag" << std::endl;
            return 29;
        }

        const pair_t pair = scfd::backend::make_value_pair( 3, 7 );
        if ( pair.first != 3 || pair.second != 7 )
        {
            std::cout << backend_name << ": FAILED value_pair host construction" << std::endl;
            return 13;
        }
        if ( !( scfd::backend::make_value_pair( 1, 4 ) < scfd::backend::make_value_pair( 2, 1 ) ) ||
             !( scfd::backend::make_value_pair( 2, 1 ) < scfd::backend::make_value_pair( 2, 3 ) ) ||
             !( scfd::backend::make_value_pair( 2, 3 ) == scfd::backend::make_value_pair( 2, 3 ) ) )
        {
            std::cout << backend_name << ": FAILED value_pair host comparison" << std::endl;
            return 21;
        }

        scfd::utils::log_std log;
        const int            init_device_with_log = runtime_t::init_device( log, 0 );
        const int            init_device_result   = runtime_t::init_device( 0 );
        int                  init_check =
            check_init_device_result<runtime_t>( backend_name, "init_device(log, 0)", init_device_with_log );
        if ( init_check != 0 )
            return 30 + init_check;
        init_check = check_init_device_result<runtime_t>( backend_name, "init_device(0)", init_device_result );
        if ( init_check != 0 )
            return 32 + init_check;
        if ( runtime_t::is_device_backend() && init_device_result != init_device_with_log )
        {
            std::cout << backend_name << ": FAILED init_device overload consistency" << std::endl;
            return 35;
        }

        runtime_t::synchronize();
        runtime_t::device_synchronize();

        const scfd::backend::device_memory_info device_info = runtime_t::get_device_memory_info();
        const scfd::backend::device_memory_info info        = runtime_t::get_memory_info();
        const scfd::backend::device_memory_info info_alias  = runtime_t::memory_info();
        int memory_check = check_device_memory_info<runtime_t>( backend_name, "get_device_memory_info()", device_info );
        if ( memory_check != 0 )
            return 40 + memory_check;
        memory_check = check_device_memory_info<runtime_t>( backend_name, "get_memory_info()", info );
        if ( memory_check != 0 )
            return 50 + memory_check;
        memory_check = check_device_memory_info<runtime_t>( backend_name, "memory_info()", info_alias );
        if ( memory_check != 0 )
            return 60 + memory_check;
        if ( info.free_bytes_known != device_info.free_bytes_known ||
             info.total_bytes_known != device_info.total_bytes_known ||
             info_alias.free_bytes_known != device_info.free_bytes_known ||
             info_alias.total_bytes_known != device_info.total_bytes_known )
        {
            std::cout << backend_name << ": FAILED memory info alias capability consistency" << std::endl;
            return 70;
        }
        if ( device_info.total_bytes_known && info.total_bytes_known && device_info.total_bytes != info.total_bytes )
        {
            std::cout << backend_name << ": FAILED memory total-bytes alias consistency" << std::endl;
            return 71;
        }

        const scfd::backend::host_memory_info host_info         = runtime_t::get_host_memory_info();
        const int                             check_host_memory = check_host_memory_info( backend_name, host_info );
        if ( check_host_memory != 0 )
            return 80 + check_host_memory;

        scfd::backend::timer_event begin;
        scfd::backend::timer_event end;
        runtime_t::synchronize();
        begin.record();
        runtime_t::synchronize();
        end.record();
        const double elapsed = end.elapsed_time( begin );
        if ( elapsed < 0.0 || elapsed != elapsed )
        {
            std::cout << backend_name << ": FAILED timer elapsed value" << std::endl;
            return 19;
        }

        array_t value_pairs;
        value_pairs.init( 4 );
        for_each_t for_each;
        for_each( fill_value_pairs_by_index<array_t>( value_pairs ), 4 );
        for_each.wait();
        runtime_t::synchronize();

        typename array_t::view_type view( value_pairs, true );
        for ( int i = 0; i < 4; ++i )
        {
            if ( view( i ).first != i || view( i ).second != i * i + 1 )
            {
                view.release( false );
                std::cout << backend_name << ": FAILED value_pair backend use" << std::endl;
                return 20;
            }
        }
        view.release( false );
    }
    catch ( const std::exception &err )
    {
        std::cout << backend_name << ": FAILED with exception: " << err.what() << std::endl;
        return 100;
    }

    std::cout << backend_name << ": PASSED" << std::endl;
    return 0;
}

}

#endif
