#ifndef __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__
#define __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__

#include <exception>
#include <iostream>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/functional/basic_ops.h>
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
    using reduce_t         = typename Backend::reduce_type;
    using sort_t           = typename Backend::sort_type;
    using unique_t         = typename Backend::unique_type;
    using exclusive_scan_t = typename Backend::exclusive_scan_type;
    using copy_t           = typename Backend::copy_type;
    using inclusive_scan_t = typename Backend::inclusive_scan_type;
    using sort_by_key_t    = typename Backend::sort_by_key_type;
    using reduce_by_key_t  = typename Backend::reduce_by_key_type;
    using set_intersection_t = typename Backend::set_intersection_type;
    using sequence_t       = typename Backend::sequence_type;

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
        if ( !std::is_same<reduce_t, scfd::backend::reduce>::value )
        {
            std::cout << backend_name << ": FAILED reduce type check" << std::endl;
            return 15;
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
        if ( !std::is_same<copy_t, scfd::backend::copy>::value )
        {
            std::cout << backend_name << ": FAILED copy type check" << std::endl;
            return 16;
        }
        if ( !std::is_same<inclusive_scan_t, scfd::backend::inclusive_scan>::value )
        {
            std::cout << backend_name << ": FAILED inclusive_scan type check" << std::endl;
            return 17;
        }
        if ( !std::is_same<sort_by_key_t, scfd::backend::sort_by_key>::value )
        {
            std::cout << backend_name << ": FAILED sort_by_key type check" << std::endl;
            return 18;
        }
        if ( !std::is_same<reduce_by_key_t, scfd::backend::reduce_by_key>::value )
        {
            std::cout << backend_name << ": FAILED reduce_by_key type check" << std::endl;
            return 24;
        }
        if ( !std::is_same<set_intersection_t, scfd::backend::set_intersection>::value )
        {
            std::cout << backend_name << ": FAILED set_intersection type check" << std::endl;
            return 25;
        }
        if ( !std::is_same<sequence_t, scfd::backend::sequence>::value )
        {
            std::cout << backend_name << ": FAILED sequence type check" << std::endl;
            return 26;
        }

        for_each_t       for_each;
        reduce_t         reduce;
        sort_t           sort;
        unique_t         unique;
        exclusive_scan_t exclusive_scan;
        copy_t           backend_copy;
        inclusive_scan_t inclusive_scan;
        sort_by_key_t    sort_by_key;
        reduce_by_key_t  reduce_by_key;
        set_intersection_t set_intersection;
        sequence_t       sequence;

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

        const int reduce_sum = reduce( 8, scan_input.raw_ptr(), 10 );
        if ( reduce_sum != 46 )
        {
            std::cout << backend_name << ": FAILED reduce sum" << std::endl;
            return 27;
        }

        const int reduce_max =
            reduce( 8, scan_input.raw_ptr(), -1, scfd::functional::maximum<int>() );
        if ( reduce_max != 8 )
        {
            std::cout << backend_name << ": FAILED reduce max" << std::endl;
            return 28;
        }

        array_t copied_values;
        copied_values.init( 8 );
        backend_copy( 8, scan_input.raw_ptr(), copied_values.raw_ptr() );
        backend_copy.wait();
        if ( !array_prefix_equal_to_expected( copied_values, host_scan_input ) )
        {
            std::cout << backend_name << ": FAILED copy" << std::endl;
            return 29;
        }

        inclusive_scan( 8, scan_input.raw_ptr(), scan_output.raw_ptr() );
        inclusive_scan.wait();
        const int expected_inclusive_scan[] = { 1, 3, 6, 10, 15, 21, 28, 36 };
        if ( !array_prefix_equal_to_expected( scan_output, expected_inclusive_scan ) )
        {
            std::cout << backend_name << ": FAILED inclusive_scan" << std::endl;
            return 30;
        }

        array_t sequence_values;
        sequence_values.init( 6 );
        sequence( 6, sequence_values.raw_ptr(), 2, 3 );
        sequence.wait();
        const int expected_sequence[] = { 2, 5, 8, 11, 14, 17 };
        if ( !array_prefix_equal_to_expected( sequence_values, expected_sequence ) )
        {
            std::cout << backend_name << ": FAILED sequence" << std::endl;
            return 31;
        }

        const int key_values[]   = { 3, 1, 2, 5, 4 };
        const int assoc_values[] = { 30, 10, 20, 50, 40 };
        array_t   keys;
        array_t   assoc;
        keys.init( 5 );
        assoc.init( 5 );
        fill_with_values( keys, key_values );
        fill_with_values( assoc, assoc_values );
        sort_by_key( 5, keys.raw_ptr(), assoc.raw_ptr() );
        sort_by_key.wait();
        const int expected_sorted_keys[]   = { 1, 2, 3, 4, 5 };
        const int expected_sorted_values[] = { 10, 20, 30, 40, 50 };
        if ( !array_prefix_equal_to_expected( keys, expected_sorted_keys ) ||
             !array_prefix_equal_to_expected( assoc, expected_sorted_values ) )
        {
            std::cout << backend_name << ": FAILED sort_by_key" << std::endl;
            return 32;
        }

        const int rbk_keys_values[] = { 1, 1, 2, 2, 2, 4 };
        const int rbk_vals_values[] = { 5, 7, 1, 2, 3, 9 };
        array_t   rbk_keys;
        array_t   rbk_vals;
        array_t   rbk_keys_out;
        array_t   rbk_vals_out;
        rbk_keys.init( 6 );
        rbk_vals.init( 6 );
        rbk_keys_out.init( 6 );
        rbk_vals_out.init( 6 );
        fill_with_values( rbk_keys, rbk_keys_values );
        fill_with_values( rbk_vals, rbk_vals_values );
        const int rbk_size =
            reduce_by_key( 6, rbk_keys.raw_ptr(), rbk_vals.raw_ptr(), rbk_keys_out.raw_ptr(), rbk_vals_out.raw_ptr() );
        reduce_by_key.wait();
        const int expected_rbk_keys[] = { 1, 2, 4 };
        const int expected_rbk_vals[] = { 12, 6, 9 };
        if ( rbk_size != 3 || !array_prefix_equal_to_expected( rbk_keys_out, expected_rbk_keys, rbk_size ) ||
             !array_prefix_equal_to_expected( rbk_vals_out, expected_rbk_vals, rbk_size ) )
        {
            std::cout << backend_name << ": FAILED reduce_by_key sum" << std::endl;
            return 33;
        }

        const int rbk_min_vals_values[] = { 8, 3, 7, 4, 9, 1 };
        fill_with_values( rbk_vals, rbk_min_vals_values );
        const int rbk_min_size = reduce_by_key(
            6, rbk_keys.raw_ptr(), rbk_vals.raw_ptr(), rbk_keys_out.raw_ptr(), rbk_vals_out.raw_ptr(),
            scfd::functional::equal_to<int>(), scfd::functional::minimum<int>()
        );
        reduce_by_key.wait();
        const int expected_rbk_min_vals[] = { 3, 4, 1 };
        if ( rbk_min_size != 3 || !array_prefix_equal_to_expected( rbk_vals_out, expected_rbk_min_vals, rbk_min_size ) )
        {
            std::cout << backend_name << ": FAILED reduce_by_key min" << std::endl;
            return 34;
        }

        const int set1_values[] = { 1, 2, 3, 5, 7 };
        const int set2_values[] = { 0, 2, 3, 4, 7 };
        array_t   set1;
        array_t   set2;
        array_t   set_result;
        set1.init( 5 );
        set2.init( 5 );
        set_result.init( 5 );
        fill_with_values( set1, set1_values );
        fill_with_values( set2, set2_values );
        const int set_size = set_intersection( 5, set1.raw_ptr(), 5, set2.raw_ptr(), set_result.raw_ptr() );
        set_intersection.wait();
        const int expected_set[] = { 2, 3, 7 };
        if ( set_size != 3 || !array_prefix_equal_to_expected( set_result, expected_set, set_size ) )
        {
            std::cout << backend_name << ": FAILED set_intersection" << std::endl;
            return 35;
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
