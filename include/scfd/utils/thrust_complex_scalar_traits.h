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

#ifndef __SCFD_UTILS_THRUST_COMPLEX_SCALAR_TRAITS_H__
#define __SCFD_UTILS_THRUST_COMPLEX_SCALAR_TRAITS_H__

#include <thrust/complex.h>
#include "scalar_traits.h"

namespace scfd
{
namespace utils
{

template <class T>
struct scalar_traits<thrust::complex<T>> : detail::complex_scalar_traits<thrust::complex<T>>
{
    __DEVICE_TAG__ static thrust::complex<T> sqrt( const thrust::complex<T> &x )
    {
        return thrust::sqrt( x );
    }
    __DEVICE_TAG__ static thrust::complex<T> pow( const thrust::complex<T> &x, const thrust::complex<T> &y )
    {
        return thrust::pow( x, y );
    }
    __DEVICE_TAG__ static thrust::complex<T> exp( const thrust::complex<T> &x )
    {
        return thrust::exp( x );
    }
    __DEVICE_TAG__ static thrust::complex<T> sin( const thrust::complex<T> &x )
    {
        return thrust::sin( x );
    }
    __DEVICE_TAG__ static thrust::complex<T> cos( const thrust::complex<T> &x )
    {
        return thrust::cos( x );
    }
};

}
}

#endif
