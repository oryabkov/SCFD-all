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

#ifndef __SCFD_UTILS_BOOST_MULTIPRECISION_SCALAR_TRAITS_H__
#define __SCFD_UTILS_BOOST_MULTIPRECISION_SCALAR_TRAITS_H__

#include <boost/multiprecision/number.hpp>
#include <boost/math/constants/constants.hpp>
#include "scalar_traits.h"

namespace scfd
{
namespace utils
{

template <class Backend, boost::multiprecision::expression_template_option Expressions>
struct scalar_traits<boost::multiprecision::number<Backend, Expressions>>
    : detail::real_scalar_traits<boost::multiprecision::number<Backend, Expressions>>
{
    using scalar_type = boost::multiprecision::number<Backend, Expressions>;
    static_assert(
        boost::multiprecision::number_category<Backend>::value != boost::multiprecision::number_kind_complex,
        "Complex multiprecision scalars require a separate specialization"
    );

    static scalar_type pi()
    {
        return boost::math::constants::pi<scalar_type>();
    }
    static scalar_type abs( const scalar_type &x )
    {
        return boost::multiprecision::abs( x );
    }
    static scalar_type sqrt( const scalar_type &x )
    {
        return boost::multiprecision::sqrt( x );
    }
    static scalar_type hypot( const scalar_type &x, const scalar_type &y )
    {
        return boost::multiprecision::hypot( x, y );
    }
    static bool is_finite( const scalar_type &x )
    {
        return boost::multiprecision::isfinite( x );
    }
    static scalar_type pow( const scalar_type &x, const scalar_type &y )
    {
        return boost::multiprecision::pow( x, y );
    }
    static scalar_type exp( const scalar_type &x )
    {
        return boost::multiprecision::exp( x );
    }
    static scalar_type sin( const scalar_type &x )
    {
        return boost::multiprecision::sin( x );
    }
    static scalar_type cos( const scalar_type &x )
    {
        return boost::multiprecision::cos( x );
    }
    static scalar_type atan2( const scalar_type &y, const scalar_type &x )
    {
        return boost::multiprecision::atan2( y, x );
    }
    static scalar_type min( const scalar_type &x, const scalar_type &y )
    {
        return x < y ? x : y;
    }
    static scalar_type max( const scalar_type &x, const scalar_type &y )
    {
        return x > y ? x : y;
    }
    static scalar_type iconst( const int &x )
    {
        return scalar_type( x );
    }
    static scalar_type fconst( const float &x )
    {
        return scalar_type( x );
    }
    static scalar_type dconst( const double &x )
    {
        return scalar_type( x );
    }
};

}
}

#endif
