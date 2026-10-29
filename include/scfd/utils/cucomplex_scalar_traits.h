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

#ifndef __SCFD_UTILS_CUCOMPLEX_SCALAR_TRAITS_H__
#define __SCFD_UTILS_CUCOMPLEX_SCALAR_TRAITS_H__

#include <cuComplex.h>
#include "scalar_traits.h"

namespace scfd
{
namespace utils
{
namespace detail
{

template <class Complex, class Component>
struct cucomplex_scalar_traits
{
    using scalar_type                = Complex;
    using component_type             = Component;
    using component_traits_type      = scalar_traits<Component>;
    using abs_type                   = typename component_traits_type::abs_type;
    static constexpr bool is_complex = true;

    __DEVICE_TAG__ static Complex make( Component re, Component im = Component( 0 ) )
    {
        return { re, im };
    }
    __DEVICE_TAG__ static Complex zero()
    {
        return make( Component( 0 ) );
    }
    __DEVICE_TAG__ static Complex from_real( Component value )
    {
        return make( value );
    }
    __DEVICE_TAG__ static Component real( const Complex &x )
    {
        return x.x;
    }
    __DEVICE_TAG__ static Component imag( const Complex &x )
    {
        return x.y;
    }
    __DEVICE_TAG__ static Complex conj( const Complex &x )
    {
        return make( x.x, -x.y );
    }
    __DEVICE_TAG__ static Complex add( const Complex &x, const Complex &y )
    {
        return make( x.x + y.x, x.y + y.y );
    }
    __DEVICE_TAG__ static Complex mul( const Complex &x, const Complex &y )
    {
        return make( x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x );
    }
    __DEVICE_TAG__ static Complex conj_mul( const Complex &x, const Complex &y )
    {
        return mul( conj( x ), y );
    }
    __DEVICE_TAG__ static abs_type abs( const Complex &x )
    {
        return component_traits_type::hypot( x.x, x.y );
    }
    __DEVICE_TAG__ static abs_type abs_sq( const Complex &x )
    {
        return x.x * x.x + x.y * x.y;
    }
    __DEVICE_TAG__ static abs_type asum_term( const Complex &x )
    {
        return component_traits_type::abs( x.x ) + component_traits_type::abs( x.y );
    }
    __DEVICE_TAG__ static bool is_finite( const Complex &x )
    {
        return component_traits_type::is_finite( x.x ) && component_traits_type::is_finite( x.y );
    }
};

}

template <>
struct scalar_traits<cuFloatComplex> : detail::cucomplex_scalar_traits<cuFloatComplex, float>
{
};

template <>
struct scalar_traits<cuDoubleComplex> : detail::cucomplex_scalar_traits<cuDoubleComplex, double>
{
};

}
}

#endif
