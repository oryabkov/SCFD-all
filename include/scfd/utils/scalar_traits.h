// Copyright © 2016,2017 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_SCALAR_TRAITS_H__
#define __SCFD_UTILS_SCALAR_TRAITS_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include "device_tag.h"

namespace scfd
{
namespace utils
{

// Unsupported scalars require an explicit specialization and its defining header.
template <class T, class Enable = void>
struct scalar_traits;

namespace detail
{

template <class T>
struct real_scalar_traits
{
    using scalar_type                = T;
    using component_type             = T;
    using abs_type                   = T;
    static constexpr bool is_complex = false;

    __DEVICE_TAG__ static T zero()
    {
        return T( 0 );
    }
    __DEVICE_TAG__ static T make( const T &value, const T & = T( 0 ) )
    {
        return value;
    }
    __DEVICE_TAG__ static T from_real( const T &value )
    {
        return value;
    }
    __DEVICE_TAG__ static T real( const T &value )
    {
        return value;
    }
    __DEVICE_TAG__ static T imag( const T & )
    {
        return T( 0 );
    }
    __DEVICE_TAG__ static T conj( const T &value )
    {
        return value;
    }
    __DEVICE_TAG__ static T add( const T &x, const T &y )
    {
        return x + y;
    }
    __DEVICE_TAG__ static T mul( const T &x, const T &y )
    {
        return x * y;
    }
    __DEVICE_TAG__ static T conj_mul( const T &x, const T &y )
    {
        return x * y;
    }
    __DEVICE_TAG__ static T abs_sq( const T &x )
    {
        return x * x;
    }
    __DEVICE_TAG__ static T sqr( const T &x )
    {
        return x * x;
    }
    __DEVICE_TAG__ static T asum_term( const T &x )
    {
        return scalar_traits<T>::abs( x );
    }
};

// Shared arithmetic for complex classes with value_type, real(), imag() and operators.
template <class Complex>
struct complex_scalar_traits
{
    using scalar_type                = Complex;
    using component_type             = typename Complex::value_type;
    using component_traits_type      = scalar_traits<component_type>;
    using abs_type                   = typename component_traits_type::abs_type;
    static constexpr bool is_complex = true;

    __DEVICE_TAG__ static Complex make( const component_type &re, const component_type &im = component_type( 0 ) )
    {
        return Complex( re, im );
    }
    __DEVICE_TAG__ static Complex zero()
    {
        return make( component_type( 0 ) );
    }
    __DEVICE_TAG__ static Complex from_real( const component_type &value )
    {
        return make( value );
    }
    __DEVICE_TAG__ static component_type real( const Complex &x )
    {
        return x.real();
    }
    __DEVICE_TAG__ static component_type imag( const Complex &x )
    {
        return x.imag();
    }
    __DEVICE_TAG__ static Complex conj( const Complex &x )
    {
        return make( x.real(), -x.imag() );
    }
    __DEVICE_TAG__ static Complex add( const Complex &x, const Complex &y )
    {
        return x + y;
    }
    __DEVICE_TAG__ static Complex mul( const Complex &x, const Complex &y )
    {
        return x * y;
    }
    __DEVICE_TAG__ static Complex conj_mul( const Complex &x, const Complex &y )
    {
        return conj( x ) * y;
    }
    __DEVICE_TAG__ static abs_type abs( const Complex &x )
    {
        return component_traits_type::hypot( x.real(), x.imag() );
    }
    __DEVICE_TAG__ static abs_type abs_sq( const Complex &x )
    {
        return component_traits_type::abs_sq( x.real() ) + component_traits_type::abs_sq( x.imag() );
    }
    // BLAS ASUM uses component absolute sums, not the complex modulus.
    __DEVICE_TAG__ static abs_type asum_term( const Complex &x )
    {
        return component_traits_type::abs( x.real() ) + component_traits_type::abs( x.imag() );
    }
    __DEVICE_TAG__ static bool is_finite( const Complex &x )
    {
        return component_traits_type::is_finite( x.real() ) && component_traits_type::is_finite( x.imag() );
    }
};

}

template <class T>
struct scalar_traits<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> : detail::real_scalar_traits<T>
{
    __DEVICE_TAG__ static T pi()
    {
        return T( 3.141592653589793238462643383279502884L );
    }
    __DEVICE_TAG__ static T sqrt( const T &x )
    {
        return std::sqrt( x );
    }
    __DEVICE_TAG__ static T abs( const T &x )
    {
        return abs_impl( x, std::is_unsigned<T>{} );
    }
    __DEVICE_TAG__ static T hypot( const T &x, const T &y )
    {
        return std::hypot( x, y );
    }
    __DEVICE_TAG__ static bool is_finite( const T &x )
    {
        return std::isfinite( x );
    }
    __DEVICE_TAG__ static T pow( const T &x, const T &y )
    {
        return std::pow( x, y );
    }
    __DEVICE_TAG__ static T exp( const T &x )
    {
        return std::exp( x );
    }
    __DEVICE_TAG__ static T sin( const T &x )
    {
        return std::sin( x );
    }
    __DEVICE_TAG__ static T cos( const T &x )
    {
        return std::cos( x );
    }
    __DEVICE_TAG__ static T atan2( const T &y, const T &x )
    {
        return std::atan2( y, x );
    }
    __DEVICE_TAG__ static T min( const T &x, const T &y )
    {
#ifdef __CUDA_ARCH__
        return std::fmin( x, y );
#else
        return std::min( x, y );
#endif
    }
    __DEVICE_TAG__ static T max( const T &x, const T &y )
    {
#ifdef __CUDA_ARCH__
        return std::fmax( x, y );
#else
        return std::max( x, y );
#endif
    }
    __DEVICE_TAG__ static T iconst( const int &x )
    {
        return T( x );
    }
    __DEVICE_TAG__ static T fconst( const float &x )
    {
        return T( x );
    }
    __DEVICE_TAG__ static T dconst( const double &x )
    {
        return T( x );
    }

private:
    __DEVICE_TAG__ static T abs_impl( const T &x, std::true_type )
    {
        return x;
    }
    __DEVICE_TAG__ static T abs_impl( const T &x, std::false_type )
    {
        return std::abs( x );
    }
};

template <class T>
struct scalar_traits<std::complex<T>>
{
    using scalar_type                = std::complex<T>;
    using component_type             = typename scalar_type::value_type;
    using component_traits_type      = scalar_traits<component_type>;
    using abs_type                   = typename component_traits_type::abs_type;
    static constexpr bool is_complex = true;

    static scalar_type make( const component_type &re, const component_type &im = component_type( 0 ) )
    {
        return scalar_type( re, im );
    }
    static scalar_type zero()
    {
        return make( component_type( 0 ) );
    }
    static scalar_type from_real( const component_type &value )
    {
        return make( value );
    }
    static component_type real( const scalar_type &x )
    {
        return x.real();
    }
    static component_type imag( const scalar_type &x )
    {
        return x.imag();
    }
    static scalar_type conj( const scalar_type &x )
    {
        return make( x.real(), -x.imag() );
    }
    static scalar_type add( const scalar_type &x, const scalar_type &y )
    {
        return x + y;
    }
    static scalar_type mul( const scalar_type &x, const scalar_type &y )
    {
        return x * y;
    }
    static scalar_type conj_mul( const scalar_type &x, const scalar_type &y )
    {
        return conj( x ) * y;
    }
    static abs_type abs( const scalar_type &x )
    {
        return component_traits_type::hypot( x.real(), x.imag() );
    }
    static abs_type abs_sq( const scalar_type &x )
    {
        return component_traits_type::abs_sq( x.real() ) + component_traits_type::abs_sq( x.imag() );
    }
    // BLAS ASUM uses component absolute sums, not the complex modulus.
    static abs_type asum_term( const scalar_type &x )
    {
        return component_traits_type::abs( x.real() ) + component_traits_type::abs( x.imag() );
    }
    static bool is_finite( const scalar_type &x )
    {
        return component_traits_type::is_finite( x.real() ) && component_traits_type::is_finite( x.imag() );
    }

    static std::complex<T> sqrt( const std::complex<T> &x )
    {
        return std::sqrt( x );
    }
    static std::complex<T> pow( const std::complex<T> &x, const std::complex<T> &y )
    {
        return std::pow( x, y );
    }
    static std::complex<T> exp( const std::complex<T> &x )
    {
        return std::exp( x );
    }
    static std::complex<T> sin( const std::complex<T> &x )
    {
        return std::sin( x );
    }
    static std::complex<T> cos( const std::complex<T> &x )
    {
        return std::cos( x );
    }
};

}
}

#endif
