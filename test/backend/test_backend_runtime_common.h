#ifndef __SCFD_TEST_BACKEND_RUNTIME_COMMON_H__
#define __SCFD_TEST_BACKEND_RUNTIME_COMMON_H__

#include <exception>
#include <iostream>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/value_pair.h>
#include <scfd/utils/device_tag.h>

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

template <class Backend>
int run_backend_runtime_tests( const char *backend_name )
{
    using runtime_t  = typename Backend::runtime_type;
    using memory_t   = typename Backend::memory_type;
    using for_each_t = typename Backend::template for_each_type<int>;
    using pair_t     = scfd::backend::value_pair<int, int>;
    using array_t    = scfd::arrays::tensor0_array_nd<pair_t, 1, memory_t>;

    try
    {
        if ( !std::is_same<Backend, scfd::backend::current>::value )
        {
            std::cout << backend_name << ": FAILED current backend type check" << std::endl;
            return 10;
        }
        if ( !std::is_same<runtime_t, scfd::backend::runtime>::value )
        {
            std::cout << backend_name << ": FAILED runtime type check" << std::endl;
            return 11;
        }
        if ( !std::is_same<typename runtime_t::timer_event_type, scfd::backend::timer_event>::value )
        {
            std::cout << backend_name << ": FAILED timer_event type check" << std::endl;
            return 12;
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

        const int init_device_result = runtime_t::init_device( 0 );
        if ( runtime_t::is_device_backend() && init_device_result < 0 )
        {
            std::cout << backend_name << ": FAILED init_device result" << std::endl;
            return 22;
        }
        if ( !runtime_t::is_device_backend() && init_device_result != -1 )
        {
            std::cout << backend_name << ": FAILED host init_device result" << std::endl;
            return 23;
        }

        runtime_t::synchronize();
        runtime_t::device_synchronize();

        const scfd::backend::device_memory_info info = runtime_t::get_memory_info();
        if ( info.free_bytes_known != runtime_t::reports_free_memory() )
        {
            std::cout << backend_name << ": FAILED memory free-bytes capability flag" << std::endl;
            return 14;
        }
        if ( info.total_bytes_known != runtime_t::reports_total_memory() )
        {
            std::cout << backend_name << ": FAILED memory total-bytes capability flag" << std::endl;
            return 15;
        }
        if ( info.free_bytes_known && info.total_bytes_known && info.free_bytes > info.total_bytes )
        {
            std::cout << backend_name << ": FAILED memory info ordering" << std::endl;
            return 16;
        }
        if ( runtime_t::reports_total_memory() && info.total_bytes == 0 )
        {
            std::cout << backend_name << ": FAILED memory total-bytes value" << std::endl;
            return 17;
        }

        const scfd::backend::device_memory_info info_alias = runtime_t::memory_info();
        if ( info_alias.free_bytes_known != info.free_bytes_known ||
             info_alias.total_bytes_known != info.total_bytes_known )
        {
            std::cout << backend_name << ": FAILED memory_info alias" << std::endl;
            return 18;
        }

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
