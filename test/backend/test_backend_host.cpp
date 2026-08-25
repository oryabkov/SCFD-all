#include <complex>
#include <iostream>
#include <type_traits>

#include "test_backend_config.h"
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/backend/backend.h>
#include <scfd/static_vec/vec.h>

namespace
{

template <class Idx, class Vec>
struct fill_imaginary_coordinate
{
    fill_imaginary_coordinate( Vec &values_ ) : values( values_ )
    {
    }

    Vec values;

    void operator()( const Idx &idx ) const
    {
        values( idx ) = { 0, static_cast<double>( idx[1] ) };
    }
};

template <class Vec>
struct fill_index
{
    fill_index( Vec &values_ ) : values( values_ )
    {
    }

    Vec values;

    void operator()( const std::size_t &idx ) const
    {
        values( idx ) = idx;
    }
};

}

int main()
{
    using value_t       = std::complex<double>;
    using backend_t     = scfd_backend_tests::expected_backend;
    using memory_t      = typename backend_t::memory_type;
    using for_each_t    = typename backend_t::template for_each_type<int>;
    using for_each_nd_t = typename backend_t::template for_each_nd_type<3>;
    using reduce_t      = typename backend_t::reduce_type;

    if ( !std::is_same<backend_t, scfd::backend::current>::value )
    {
        std::cout << "FAILED BACKEND TYPE CHECK" << std::endl;
        return 10;
    }

    using array_t  = scfd::arrays::tensor0_array_nd<value_t, 1, memory_t>;
    using array3_t = scfd::arrays::tensor0_array_nd<value_t, 3, memory_t>;
    using idx3_t   = scfd::static_vec::vec<int, 3>;
    using rect_t   = scfd::static_vec::rect<int, 3>;

    for_each_t    for_each;
    for_each_nd_t for_each_nd;
    reduce_t      reduce;

    const int size_x = 10;
    const int size_y = 10;
    const int size_z = 10;
    const int size   = size_x * size_y * size_z;

    array3_t values3;
    values3.init( idx3_t( size_x, size_y, size_z ) );
    const rect_t range( idx3_t( 0, 0, 0 ), idx3_t( size_x, size_y, size_z ) );
    for_each_nd( fill_imaginary_coordinate<idx3_t, array3_t>( values3 ), range );

    auto    result    = reduce( size, values3.raw_ptr(), value_t( 0, 0 ) );
    value_t reference = value_t( 0, 10 * 10 * ( 9 * ( 9 + 1 ) ) / 2 );
    reduce.wait();
    double difference = std::norm( result - reference );
    std::cout << "res = " << result << ", res_ref = " << reference << ", ||diff|| = " << difference << std::endl;

    array_t values1;
    values1.init( size );
    for_each( fill_index<array_t>( values1 ), size );
    result    = reduce( size, values1.raw_ptr(), value_t( 0, 0 ) );
    reference = value_t( ( size - 1 ) * size * 0.5, 0 );
    reduce.wait();
    difference += std::norm( result - reference );
    std::cout << "res = " << result << ", res_ref = " << reference << ", ||diff|| = " << difference << std::endl;

    if ( difference > 1.0e-12 )
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }

    std::cout << scfd_backend_tests::expected_backend_configuration_name() << ": PASSED" << std::endl;
    return 0;
}
