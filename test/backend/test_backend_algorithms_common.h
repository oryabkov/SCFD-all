#ifndef __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__
#define __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__

#include <exception>
#include <iostream>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/utils/device_tag.h>

namespace scfd_backend_tests
{

template <class Array>
struct fill_by_index
{
    fill_by_index( const Array &data_ ) : data( data_ )
    {
    }

    Array data;

    __DEVICE_TAG__ void operator()( const int &idx ) const
    {
        data( idx ) = idx * idx - 3;
    }
};

}

#ifdef PLATFORM_SYCL
template <class Array>
struct sycl::is_device_copyable<scfd_backend_tests::fill_by_index<Array>> : std::true_type
{
};
#endif

namespace scfd_backend_tests
{

template <class Array, int N>
void fill_with_values( Array &array, const int ( &values )[N] )
{
    typename Array::view_type view( array, false );
    for ( int i = 0; i < N; ++i )
    {
        view( i ) = values[i];
    }
    view.release( true );
}

template <class Array, int N>
bool array_prefix_equal_to_expected( const Array &array, const int ( &expected )[N], int size = N )
{
    typename Array::view_type view( array, true );
    bool                      result = true;
    for ( int i = 0; i < size; ++i )
    {
        if ( view( i ) != expected[i] )
        {
            result = false;
            break;
        }
    }
    view.release( false );
    return result;
}

template <class Backend>
int run_backend_algorithm_tests( const char *backend_name )
{
    using memory_t         = typename Backend::memory_type;
    using array_t          = scfd::arrays::tensor0_array_nd<int, 1, memory_t>;
    using for_each_t       = typename Backend::template for_each_type<int>;
    using sort_t           = typename Backend::sort_type;
    using unique_t         = typename Backend::unique_type;
    using exclusive_scan_t = typename Backend::exclusive_scan_type;

    try
    {
        if ( !std::is_same<Backend, scfd::backend::current>::value )
        {
            std::cout << backend_name << ": FAILED current backend type check" << std::endl;
            return 10;
        }
        if ( !std::is_same<sort_t, scfd::backend::sort>::value )
        {
            std::cout << backend_name << ": FAILED sort type check" << std::endl;
            return 11;
        }
        if ( !std::is_same<unique_t, scfd::backend::unique>::value )
        {
            std::cout << backend_name << ": FAILED unique type check" << std::endl;
            return 12;
        }
        if ( !std::is_same<exclusive_scan_t, scfd::backend::exclusive_scan>::value )
        {
            std::cout << backend_name << ": FAILED exclusive_scan type check" << std::endl;
            return 13;
        }
        if ( !std::is_same<for_each_t, scfd::backend::for_each<int>>::value )
        {
            std::cout << backend_name << ": FAILED for_each type check" << std::endl;
            return 14;
        }

        for_each_t       for_each;
        sort_t           sort;
        unique_t         unique;
        exclusive_scan_t exclusive_scan;

        array_t for_each_values;
        for_each_values.init( 5 );
        for_each( fill_by_index<array_t>( for_each_values ), 5 );
        for_each.wait();

        const int expected_for_each[] = { -3, -2, 1, 6, 13 };
        if ( !array_prefix_equal_to_expected( for_each_values, expected_for_each ) )
        {
            std::cout << backend_name << ": FAILED for_each" << std::endl;
            return 19;
        }

        const int host_values[] = { 4, 2, 2, 1, 3, 3, 3 };
        array_t   values;
        values.init( 7 );
        fill_with_values( values, host_values );

        sort( 7, values.raw_ptr() );
        sort.wait();

        const int expected_sorted[] = { 1, 2, 2, 3, 3, 3, 4 };
        if ( !array_prefix_equal_to_expected( values, expected_sorted ) )
        {
            std::cout << backend_name << ": FAILED sort" << std::endl;
            return 20;
        }

        const int unique_size = unique( 7, values.raw_ptr() );
        unique.wait();

        const int expected_unique[] = { 1, 2, 3, 4 };
        if ( unique_size != 4 || !array_prefix_equal_to_expected( values, expected_unique, unique_size ) )
        {
            std::cout << backend_name << ": FAILED unique" << std::endl;
            return 21;
        }

        const int host_scan_input[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        array_t   scan_input;
        array_t   scan_output;
        scan_input.init( 8 );
        scan_output.init( 8 );
        fill_with_values( scan_input, host_scan_input );

        exclusive_scan( 8, scan_input.raw_ptr(), scan_output.raw_ptr(), 10 );
        exclusive_scan.wait();

        const int expected_scan[] = { 10, 11, 13, 16, 20, 25, 31, 38 };
        if ( !array_prefix_equal_to_expected( scan_output, expected_scan ) )
        {
            std::cout << backend_name << ": FAILED exclusive_scan" << std::endl;
            return 22;
        }

        const int host_in_place[] = { 1, 2, 3, 4 };
        array_t   in_place;
        in_place.init( 4 );
        fill_with_values( in_place, host_in_place );

        exclusive_scan( 4, in_place.raw_ptr(), in_place.raw_ptr(), 0 );
        exclusive_scan.wait();

        const int expected_in_place[] = { 0, 1, 3, 6 };
        if ( !array_prefix_equal_to_expected( in_place, expected_in_place ) )
        {
            std::cout << backend_name << ": FAILED in-place exclusive_scan" << std::endl;
            return 23;
        }
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
