// Copyright (C) 2026 SCFD contributors

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <scfd/utils/scalar_traits.h>

#ifdef SCFD_TEST_WITH_BOOST
#    include <boost/multiprecision/cpp_bin_float.hpp>
#    include <boost/multiprecision/cpp_dec_float.hpp>
#    include <scfd/utils/boost_multiprecision_scalar_traits.h>
#endif
#ifdef SCFD_TEST_BOOST_ISOLATION
#    include <boost/multiprecision/cpp_bin_float.hpp>
#endif

struct dual
{
    double value, derivative;
};

namespace scfd
{
namespace utils
{
template <>
struct scalar_traits<dual>
{
    using abs_type = dual;
    static dual abs( const dual &x )
    {
        return x.value < 0 ? dual{ -x.value, -x.derivative } : x;
    }
    static dual abs_sq( const dual &x )
    {
        return { x.value * x.value, 2 * x.value * x.derivative };
    }
};
}
}

namespace
{
unsigned int checks = 0;
void         require( bool passed )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( "Scalar traits regression failed" );
}

template <class T, class = void>
struct has_traits : std::false_type
{
};
template <class T>
struct has_traits<T, std::void_t<typename scfd::utils::scalar_traits<T>::abs_type>> : std::true_type
{
};
struct unsupported;
static_assert( !has_traits<unsupported>::value );
#ifdef SCFD_TEST_BOOST_ISOLATION
static_assert(
    !has_traits<boost::multiprecision::cpp_bin_float_100>::value,
    "Boost support must require its explicit traits header"
);
#endif

template <class T>
void test_real()
{
    using traits_type = scfd::utils::scalar_traits<T>;
    static_assert( std::is_same_v<typename traits_type::abs_type, T> );
    static_assert( std::is_same_v<decltype( traits_type::abs( T( 3 ) ) ), T> );
    require( traits_type::zero() == T( 0 ) && !traits_type::is_complex );
    require( traits_type::abs( T( -3 ) ) == T( 3 ) );
    require( traits_type::abs_sq( T( -3 ) ) == T( 9 ) );
    require( traits_type::asum_term( T( -3 ) ) == T( 3 ) );
    require( traits_type::conj_mul( T( -3 ), T( 2 ) ) == T( -6 ) );
    require( traits_type::sqrt( T( 4 ) ) == T( 2 ) );
    require( traits_type::hypot( T( 3 ), T( 4 ) ) == T( 5 ) );
    require( traits_type::is_finite( std::numeric_limits<T>::max() ) );
    require( !traits_type::is_finite( std::numeric_limits<T>::infinity() ) );
    require( !traits_type::is_finite( std::numeric_limits<T>::quiet_NaN() ) );
    const T huge   = std::numeric_limits<T>::max() / 4;
    const T radius = traits_type::hypot( huge, huge );
    require( radius > huge && traits_type::is_finite( radius ) );
    require( traits_type::sin( T( 0 ) ) == T( 0 ) && traits_type::cos( T( 0 ) ) == T( 1 ) );
}

template <class T>
void test_complex()
{
    using traits_type = scfd::utils::scalar_traits<std::complex<T>>;
    using abs_type    = typename traits_type::abs_type;
    static_assert( std::is_same_v<abs_type, typename scfd::utils::scalar_traits<T>::abs_type> );
    const auto x = traits_type::make( T( 3 ), T( 4 ) );
    const auto y = traits_type::make( T( -2 ), T( 1 ) );
    require( traits_type::abs( x ) == abs_type( 5 ) );
    require( traits_type::abs_sq( x ) == abs_type( 25 ) );
    require( traits_type::asum_term( x ) == abs_type( 7 ) );
    const auto dot = traits_type::conj_mul( x, y );
    require( traits_type::real( dot ) == T( -2 ) && traits_type::imag( dot ) == T( 11 ) );
    require( traits_type::abs_sq( traits_type::zero() ) == abs_type( 0 ) );
    require( traits_type::is_finite( x ) );
    require( !traits_type::is_finite( traits_type::make( T( 0 ), std::numeric_limits<T>::infinity() ) ) );
    require( !traits_type::is_finite( traits_type::make( std::numeric_limits<T>::quiet_NaN(), T( 0 ) ) ) );
    const T    huge   = std::numeric_limits<T>::max() / 4;
    const auto radius = traits_type::abs( traits_type::make( huge, huge ) );
    require( radius > huge && scfd::utils::scalar_traits<abs_type>::is_finite( radius ) );
}

#ifdef SCFD_TEST_WITH_BOOST
template <class T>
void test_precision()
{
    using traits_type = scfd::utils::scalar_traits<T>;
    const T tiny( "1e-60" );
    const T value = T( 1 ) + tiny;
    require( traits_type::abs( -value ) == value && traits_type::abs( -value ) > T( 1 ) );
    require( traits_type::abs_sq( value ) > T( 1 ) );
    const T huge( "1e1000" );
    require( traits_type::hypot( huge, huge ) > huge );
    require( traits_type::is_finite( traits_type::hypot( huge, huge ) ) );
    using complex_traits_type = scfd::utils::scalar_traits<std::complex<T>>;
    require( complex_traits_type::abs( complex_traits_type::make( T( 0 ), value ) ) == value );
}
#endif
}

int main()
{
    try
    {
        test_real<float>();
        test_real<double>();
        test_real<long double>();
        test_complex<float>();
        test_complex<double>();
        test_complex<long double>();
        require( scfd::utils::scalar_traits<unsigned int>::abs( 3u ) == 3u );
        const auto magnitude = scfd::utils::scalar_traits<dual>::abs( { -3, 2 } );
        require( magnitude.value == 3 && magnitude.derivative == -2 );
        const auto square = scfd::utils::scalar_traits<dual>::abs_sq( { -3, 2 } );
        require( square.value == 9 && square.derivative == -12 );
#ifdef SCFD_TEST_WITH_BOOST
        using namespace boost::multiprecision;
        test_real<cpp_bin_float_100>();
        test_real<cpp_dec_float_100>();
        test_complex<cpp_bin_float_100>();
        test_precision<cpp_bin_float_100>();
        test_precision<number<cpp_bin_float<100>, et_on>>();
        test_precision<cpp_dec_float_100>();
#endif
        std::printf( "Scalar traits: %u checks, 0 failures\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "%s (check %u)\n", error.what(), checks );
        return 1;
    }
}
