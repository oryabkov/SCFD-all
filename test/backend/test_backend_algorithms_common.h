#ifndef __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__
#define __SCFD_TEST_BACKEND_ALGORITHMS_COMMON_H__

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <type_traits>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/functional/basic_ops.h>
#include <scfd/backend/value_pair.h>
#include <scfd/utils/device_tag.h>

namespace scfd_backend_tests
{

template <class Array, class Ordinal>
struct fill_by_index
{
    fill_by_index( const Array &data_ ) : data( data_ )
    {
    }

    Array data;

    __DEVICE_TAG__ void operator()( const Ordinal &idx ) const
    {
        const int value = static_cast<int>( idx );
        data( idx )     = value * value - 3;
    }
};

template <class Ordinal>
struct record_offset_index
{
    Ordinal *data;
    Ordinal  offset;

    __DEVICE_TAG__ void operator()( const Ordinal &idx ) const
    {
        data[idx - offset] = idx;
    }
};

template <class Ordinal>
struct record_offset_coordinate
{
    Ordinal *data;
    Ordinal  offset;

    __DEVICE_TAG__ void operator()( const scfd::static_vec::vec<Ordinal, 2> &idx ) const
    {
        data[( idx[0] - offset ) * 2 + idx[1] - offset] = idx[0] + idx[1];
    }
};

}

#ifdef PLATFORM_SYCL
template <class Array, class Ordinal>
struct sycl::is_device_copyable<scfd_backend_tests::fill_by_index<Array, Ordinal>> : std::true_type
{
};
#endif

namespace scfd_backend_tests
{

template <class Array, int N, class Ordinal = int>
bool array_prefix_equal_to_expected( const Array &array, const int ( &expected )[N], Ordinal size = N )
{
    typename Array::view_type view( array, true );
    bool                      result = true;
    for ( Ordinal i = 0; i < size; ++i )
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

template <class Array, class T, int N, class Ordinal = int>
bool array_prefix_equal_to_typed_expected( const Array &array, const T ( &expected )[N], Ordinal size = N )
{
    typename Array::view_type view( array, true );
    bool                      result = true;
    for ( Ordinal i = 0; i < size; ++i )
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
    using ordinal_t          = typename Backend::ordinal_type;
    using memory_t           = typename Backend::memory_type;
    using array_t            = scfd::arrays::tensor0_array_nd<int, 1, memory_t>;
    using pair_t             = scfd::backend::value_pair<int, int>;
    using pair_array_t       = scfd::arrays::tensor0_array_nd<pair_t, 1, memory_t>;
    using for_each_t         = typename Backend::for_each_type;
    using for_each_nd_t      = typename Backend::template for_each_nd_type<2>;
    using reduce_t           = typename Backend::reduce_type;
    using sort_t             = typename Backend::sort_type;
    using unique_t           = typename Backend::unique_type;
    using exclusive_scan_t   = typename Backend::exclusive_scan_type;
    using copy_t             = typename Backend::copy_type;
    using inclusive_scan_t   = typename Backend::inclusive_scan_type;
    using sort_by_key_t      = typename Backend::sort_by_key_type;
    using reduce_by_key_t    = typename Backend::reduce_by_key_type;
    using set_intersection_t = typename Backend::set_intersection_type;
    using sequence_t         = typename Backend::sequence_type;
    using count_by_key_t     = typename Backend::count_by_key_type;

#if defined( PLATFORM_SERIAL_CPU )
#    define SCFD_TEST_ALGORITHM_IMPL serial
    using expected_for_each_t    = scfd::for_each::serial_cpu<ordinal_t>;
    using expected_for_each_nd_t = scfd::for_each::serial_cpu_nd<2, ordinal_t>;
    using expected_copy_t        = scfd::copy::serial<ordinal_t>;
#elif defined( PLATFORM_OMP )
#    define SCFD_TEST_ALGORITHM_IMPL omp
    using expected_for_each_t    = scfd::for_each::openmp<ordinal_t>;
    using expected_for_each_nd_t = scfd::for_each::openmp_nd<2, ordinal_t>;
    using expected_copy_t        = scfd::copy::omp<ordinal_t>;
#elif defined( PLATFORM_CUDA )
#    define SCFD_TEST_ALGORITHM_IMPL thrust
    using expected_for_each_t    = scfd::for_each::cuda<ordinal_t>;
    using expected_for_each_nd_t = scfd::for_each::cuda_nd<2, ordinal_t>;
    using expected_copy_t        = scfd::copy::cuda<ordinal_t>;
#elif defined( PLATFORM_HIP )
#    define SCFD_TEST_ALGORITHM_IMPL thrust
    using expected_for_each_t    = scfd::for_each::hip<ordinal_t>;
    using expected_for_each_nd_t = scfd::for_each::hip_nd<2, ordinal_t>;
    using expected_copy_t        = scfd::copy::hip<ordinal_t>;
#elif defined( PLATFORM_SYCL )
#    define SCFD_TEST_ALGORITHM_IMPL sycl
    using expected_for_each_t    = scfd::for_each::sycl_<ordinal_t>;
    using expected_for_each_nd_t = scfd::for_each::sycl_nd<2, ordinal_t>;
    using expected_copy_t        = scfd::copy::sycl<ordinal_t>;
#endif

    static_assert( std::is_same<Backend, scfd::backend::current<ordinal_t>>::value, "current backend type" );
    static_assert( std::is_same<for_each_t, expected_for_each_t>::value, "for_each ordinal propagation" );
    static_assert( std::is_same<for_each_nd_t, expected_for_each_nd_t>::value, "for_each_nd ordinal propagation" );
    static_assert( std::is_same<copy_t, expected_copy_t>::value, "copy ordinal propagation" );
    static_assert( std::is_same<for_each_t, scfd::backend::for_each<ordinal_t>>::value, "for_each shortcut" );
    static_assert(
        std::is_same<for_each_nd_t, scfd::backend::for_each_nd<2, ordinal_t>>::value, "for_each_nd shortcut"
    );
    static_assert( std::is_same<copy_t, scfd::backend::copy<ordinal_t>>::value, "copy shortcut" );

#define SCFD_TEST_ALGORITHM_ORDINAL( operation )                                                                       \
    static_assert(                                                                                                     \
        std::is_same<operation##_t, scfd::operation::SCFD_TEST_ALGORITHM_IMPL<ordinal_t>>::value,                      \
        #operation " ordinal propagation"                                                                              \
    );                                                                                                                 \
    static_assert( std::is_same<operation##_t, scfd::backend::operation<ordinal_t>>::value, #operation " shortcut" )

    SCFD_TEST_ALGORITHM_ORDINAL( reduce );
    SCFD_TEST_ALGORITHM_ORDINAL( sort );
    SCFD_TEST_ALGORITHM_ORDINAL( unique );
    SCFD_TEST_ALGORITHM_ORDINAL( exclusive_scan );
    SCFD_TEST_ALGORITHM_ORDINAL( inclusive_scan );
    SCFD_TEST_ALGORITHM_ORDINAL( sort_by_key );
    SCFD_TEST_ALGORITHM_ORDINAL( reduce_by_key );
    SCFD_TEST_ALGORITHM_ORDINAL( set_intersection );
    SCFD_TEST_ALGORITHM_ORDINAL( sequence );
    SCFD_TEST_ALGORITHM_ORDINAL( count_by_key );

#undef SCFD_TEST_ALGORITHM_ORDINAL
#undef SCFD_TEST_ALGORITHM_IMPL

    try
    {
        for_each_t         for_each;
        reduce_t           reduce;
        sort_t             sort;
        unique_t           unique;
        exclusive_scan_t   exclusive_scan;
        copy_t             backend_copy;
        inclusive_scan_t   inclusive_scan;
        sort_by_key_t      sort_by_key;
        reduce_by_key_t    reduce_by_key;
        set_intersection_t set_intersection;
        sequence_t         sequence;
        count_by_key_t     count_by_key;

        array_t for_each_values;
        for_each_values.init( 5 );
        for_each( fill_by_index<array_t, ordinal_t>( for_each_values ), ordinal_t( 5 ) );
        for_each.wait();

        const int expected_for_each[] = { -3, -2, 1, 6, 13 };
        if ( !array_prefix_equal_to_expected( for_each_values, expected_for_each ) )
        {
            std::cout << backend_name << ": FAILED for_each" << std::endl;
            return 19;
        }

        // Exercise wide logical indices without allocating a large array. The callbacks
        // use raw pointers so the array's independently configured ordinal does not constrain the range.
        const ordinal_t offset =
            std::numeric_limits<ordinal_t>::digits > std::numeric_limits<int>::digits
                ? static_cast<ordinal_t>( static_cast<std::uintmax_t>( std::numeric_limits<int>::max() ) + 17 )
                : ordinal_t( 17 );
        using ordinal_array_t         = scfd::arrays::tensor0_array_nd<ordinal_t, 1, memory_t>;
        ordinal_array_t offset_values = { ordinal_t( -1 ), ordinal_t( -1 ), ordinal_t( -1 ), ordinal_t( -1 ) };
        for_each( record_offset_index<ordinal_t>{ offset_values.raw_ptr(), offset }, offset, offset + 4 );
        for_each.wait();
        const ordinal_t expected_offset[] = { offset, offset + 1, offset + 2, offset + 3 };
        if ( !array_prefix_equal_to_typed_expected( offset_values, expected_offset ) )
        {
            std::cout << backend_name << ": FAILED for_each offset range" << std::endl;
            return 42;
        }
        using idx_t  = scfd::static_vec::vec<ordinal_t, 2>;
        using rect_t = scfd::static_vec::rect<ordinal_t, 2>;
        for_each_nd_t for_each_nd;
        for_each_nd(
            record_offset_coordinate<ordinal_t>{ offset_values.raw_ptr(), offset },
            rect_t( idx_t( offset, offset ), idx_t( offset + 2, offset + 2 ) )
        );
        for_each_nd.wait();
        const ordinal_t expected_coordinates[] = { 2 * offset, 2 * offset + 1, 2 * offset + 1, 2 * offset + 2 };
        if ( !array_prefix_equal_to_typed_expected( offset_values, expected_coordinates ) )
        {
            std::cout << backend_name << ": FAILED for_each_nd offset range" << std::endl;
            return 43;
        }

        array_t values = { 4, 2, 2, 1, 3, 3, 3 };

        sort( 7, values.raw_ptr() );
        sort.wait();

        const int expected_sorted[] = { 1, 2, 2, 3, 3, 3, 4 };
        if ( !array_prefix_equal_to_expected( values, expected_sorted ) )
        {
            std::cout << backend_name << ": FAILED sort" << std::endl;
            return 20;
        }

        const auto unique_size = unique( 7, values.raw_ptr() );
        static_assert( std::is_same<decltype( unique_size ), const ordinal_t>::value, "unique returns the ordinal" );
        unique.wait();

        const int expected_unique[] = { 1, 2, 3, 4 };
        if ( unique_size != 4 || !array_prefix_equal_to_expected( values, expected_unique, unique_size ) )
        {
            std::cout << backend_name << ": FAILED unique" << std::endl;
            return 21;
        }

        array_t scan_input = { 1, 2, 3, 4, 5, 6, 7, 8 };
        array_t scan_output;
        scan_output.init( 8 );

        exclusive_scan( 8, scan_input.raw_ptr(), scan_output.raw_ptr(), 10 );
        exclusive_scan.wait();

        const int expected_scan[] = { 10, 11, 13, 16, 20, 25, 31, 38 };
        if ( !array_prefix_equal_to_expected( scan_output, expected_scan ) )
        {
            std::cout << backend_name << ": FAILED exclusive_scan" << std::endl;
            return 22;
        }

        array_t in_place = { 1, 2, 3, 4 };

        exclusive_scan( 4, in_place.raw_ptr(), in_place.raw_ptr(), 0 );
        exclusive_scan.wait();

        const int expected_in_place[] = { 0, 1, 3, 6 };
        if ( !array_prefix_equal_to_expected( in_place, expected_in_place ) )
        {
            std::cout << backend_name << ": FAILED in-place exclusive_scan" << std::endl;
            return 23;
        }

        const auto reduce_sum = reduce( 8, scan_input.raw_ptr(), 10 );
        static_assert( std::is_same<decltype( reduce_sum ), const int>::value, "reduce returns the data type" );
        if ( reduce_sum != 46 )
        {
            std::cout << backend_name << ": FAILED reduce sum" << std::endl;
            return 27;
        }

        const int reduce_max = reduce( 8, scan_input.raw_ptr(), -1, scfd::functional::maximum<int>() );
        if ( reduce_max != 8 )
        {
            std::cout << backend_name << ": FAILED reduce max" << std::endl;
            return 28;
        }

        array_t copied_values;
        copied_values.init( 8 );
        backend_copy( 8, scan_input.raw_ptr(), copied_values.raw_ptr() );
        backend_copy.wait();
        const int expected_copied_values[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        if ( !array_prefix_equal_to_expected( copied_values, expected_copied_values ) )
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

        array_t keys  = { 3, 1, 2, 5, 4 };
        array_t assoc = { 30, 10, 20, 50, 40 };
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

        array_t rbk_keys = { 1, 1, 2, 2, 2, 4 };
        array_t rbk_vals = { 5, 7, 1, 2, 3, 9 };
        array_t rbk_keys_out;
        array_t rbk_vals_out;
        rbk_keys_out.init( 6 );
        rbk_vals_out.init( 6 );
        const auto rbk_size =
            reduce_by_key( 6, rbk_keys.raw_ptr(), rbk_vals.raw_ptr(), rbk_keys_out.raw_ptr(), rbk_vals_out.raw_ptr() );
        static_assert(
            std::is_same<decltype( rbk_size ), const ordinal_t>::value, "reduce_by_key returns the ordinal"
        );
        reduce_by_key.wait();
        const int expected_rbk_keys[] = { 1, 2, 4 };
        const int expected_rbk_vals[] = { 12, 6, 9 };
        if ( rbk_size != 3 || !array_prefix_equal_to_expected( rbk_keys_out, expected_rbk_keys, rbk_size ) ||
             !array_prefix_equal_to_expected( rbk_vals_out, expected_rbk_vals, rbk_size ) )
        {
            std::cout << backend_name << ": FAILED reduce_by_key sum" << std::endl;
            return 33;
        }

        rbk_vals                     = { 8, 3, 7, 4, 9, 1 };
        const ordinal_t rbk_min_size = reduce_by_key(
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

        const int rbk_stress_size   = 257;
        const int rbk_stress_group  = 37;
        const int rbk_stress_groups = ( rbk_stress_size + rbk_stress_group - 1 ) / rbk_stress_group;
        array_t   rbk_stress_keys;
        array_t   rbk_stress_vals;
        array_t   rbk_stress_keys_out;
        array_t   rbk_stress_vals_out;
        rbk_stress_keys.init( rbk_stress_size );
        rbk_stress_vals.init( rbk_stress_size );
        rbk_stress_keys_out.init( rbk_stress_groups );
        rbk_stress_vals_out.init( rbk_stress_groups );
        {
            typename array_t::view_type keys_view( rbk_stress_keys, false );
            typename array_t::view_type vals_view( rbk_stress_vals, false );
            for ( int i = 0; i < rbk_stress_size; ++i )
            {
                keys_view( i ) = i / rbk_stress_group;
                vals_view( i ) = 1;
            }
            keys_view.release( true );
            vals_view.release( true );
        }
        const ordinal_t rbk_stress_out_size = reduce_by_key(
            rbk_stress_size, rbk_stress_keys.raw_ptr(), rbk_stress_vals.raw_ptr(), rbk_stress_keys_out.raw_ptr(),
            rbk_stress_vals_out.raw_ptr()
        );
        reduce_by_key.wait();
        bool rbk_stress_ok = rbk_stress_out_size == rbk_stress_groups;
        {
            typename array_t::view_type keys_view( rbk_stress_keys_out, true );
            typename array_t::view_type vals_view( rbk_stress_vals_out, true );
            for ( int i = 0; i < rbk_stress_groups; ++i )
            {
                const int first_index    = i * rbk_stress_group;
                const int expected_count = std::min( rbk_stress_group, rbk_stress_size - first_index );
                if ( keys_view( i ) != i || vals_view( i ) != expected_count )
                {
                    rbk_stress_ok = false;
                    break;
                }
            }
            keys_view.release( false );
            vals_view.release( false );
        }
        if ( !rbk_stress_ok )
        {
            std::cout << backend_name << ": FAILED reduce_by_key chunk stress" << std::endl;
            return 36;
        }

        array_t set1 = { 1, 2, 3, 5, 7 };
        array_t set2 = { 0, 2, 3, 4, 7 };
        array_t set_result;
        set_result.init( 5 );
        const auto set_size = set_intersection( 5, set1.raw_ptr(), 5, set2.raw_ptr(), set_result.raw_ptr() );
        static_assert(
            std::is_same<decltype( set_size ), const ordinal_t>::value, "set_intersection returns the ordinal"
        );
        set_intersection.wait();
        const int expected_set[] = { 2, 3, 7 };
        if ( set_size != 3 || !array_prefix_equal_to_expected( set_result, expected_set, set_size ) )
        {
            std::cout << backend_name << ": FAILED set_intersection" << std::endl;
            return 35;
        }

        const int set_stress_unique = 32;
        const int set_stress_size1  = set_stress_unique * 3;
        const int set_stress_size2  = set_stress_unique * 2;
        const int set_stress_size   = set_stress_unique * 2;
        array_t   set_stress1;
        array_t   set_stress2;
        array_t   set_stress_result;
        set_stress1.init( set_stress_size1 );
        set_stress2.init( set_stress_size2 );
        set_stress_result.init( set_stress_size );
        {
            typename array_t::view_type set1_view( set_stress1, false );
            typename array_t::view_type set2_view( set_stress2, false );
            for ( int i = 0; i < set_stress_size1; ++i )
                set1_view( i ) = i / 3;
            for ( int i = 0; i < set_stress_size2; ++i )
                set2_view( i ) = i / 2;
            set1_view.release( true );
            set2_view.release( true );
        }
        const ordinal_t set_stress_out_size = set_intersection(
            set_stress_size1, set_stress1.raw_ptr(), set_stress_size2, set_stress2.raw_ptr(),
            set_stress_result.raw_ptr()
        );
        set_intersection.wait();
        bool set_stress_ok = set_stress_out_size == set_stress_size;
        {
            typename array_t::view_type result_view( set_stress_result, true );
            for ( int i = 0; i < set_stress_unique; ++i )
            {
                if ( result_view( 2 * i ) != i || result_view( 2 * i + 1 ) != i )
                {
                    set_stress_ok = false;
                    break;
                }
            }
            result_view.release( false );
        }
        if ( !set_stress_ok )
        {
            std::cout << backend_name << ": FAILED set_intersection duplicate stress" << std::endl;
            return 37;
        }

        array_t count_keys = { 1, 1, 1, 3, 4, 4, 8 };
        array_t count_keys_out;
        using count_array_t = scfd::arrays::tensor0_array_nd<long long, 1, memory_t>;
        count_array_t count_values_out;
        count_keys_out.init( 7 );
        count_values_out.init( 7 );
        const auto count_by_key_size =
            count_by_key( 7, count_keys.raw_ptr(), count_keys_out.raw_ptr(), count_values_out.raw_ptr() );
        static_assert(
            std::is_same<decltype( count_by_key_size ), const ordinal_t>::value, "count_by_key returns the ordinal"
        );
        count_by_key.wait();
        const int       expected_count_keys[] = { 1, 3, 4, 8 };
        const long long expected_count_vals[] = { 3, 1, 2, 1 };
        if ( count_by_key_size != 4 ||
             !array_prefix_equal_to_expected( count_keys_out, expected_count_keys, count_by_key_size ) ||
             !array_prefix_equal_to_typed_expected( count_values_out, expected_count_vals, count_by_key_size ) )
        {
            std::cout << backend_name << ": FAILED count_by_key" << std::endl;
            return 39;
        }

        pair_array_t pair_data = { pair_t( 2, 3 ), pair_t( 1, 4 ), pair_t( 2, 1 ), pair_t( 1, 4 ), pair_t( 2, 1 ) };
        sort( 5, pair_data.raw_ptr() );
        sort.wait();
        const pair_t expected_pair_sorted[] = {
            pair_t( 1, 4 ), pair_t( 1, 4 ), pair_t( 2, 1 ), pair_t( 2, 1 ), pair_t( 2, 3 )
        };
        if ( !array_prefix_equal_to_typed_expected( pair_data, expected_pair_sorted ) )
        {
            std::cout << backend_name << ": FAILED value_pair sort" << std::endl;
            return 40;
        }
        const ordinal_t pair_unique_size = unique( 5, pair_data.raw_ptr() );
        unique.wait();
        const pair_t expected_pair_unique[] = { pair_t( 1, 4 ), pair_t( 2, 1 ), pair_t( 2, 3 ) };
        if ( pair_unique_size != 3 ||
             !array_prefix_equal_to_typed_expected( pair_data, expected_pair_unique, pair_unique_size ) )
        {
            std::cout << backend_name << ": FAILED value_pair unique" << std::endl;
            return 41;
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
