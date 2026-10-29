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

#include <cstdio>
#include <limits>
#include <type_traits>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cufft.h>
#include <scfd/utils/thrust_complex_scalar_traits.h>
#include <scfd/utils/cucomplex_scalar_traits.h>

namespace
{
template <class T>
__device__ bool check_real()
{
    using traits_type = scfd::utils::scalar_traits<T>;
    return traits_type::abs( T( -3 ) ) == T( 3 ) && traits_type::abs_sq( T( -3 ) ) == T( 9 ) &&
           traits_type::asum_term( T( -3 ) ) == T( 3 ) && traits_type::hypot( T( 3 ), T( 4 ) ) == T( 5 ) &&
           traits_type::sqrt( T( 4 ) ) == T( 2 ) && traits_type::is_finite( T( 3 ) ) &&
           !traits_type::is_finite( T( CUDART_INF ) );
}

template <class Complex>
__device__ bool check_complex()
{
    using traits_type = scfd::utils::scalar_traits<Complex>;
    using T           = typename traits_type::abs_type;
    const auto x      = traits_type::make( T( 3 ), T( 4 ) );
    const auto y      = traits_type::make( T( -2 ), T( 1 ) );
    const auto dot    = traits_type::conj_mul( x, y );
    return traits_type::abs( x ) == T( 5 ) && traits_type::abs_sq( x ) == T( 25 ) &&
           traits_type::asum_term( x ) == T( 7 ) && traits_type::real( dot ) == T( -2 ) &&
           traits_type::imag( dot ) == T( 11 ) && traits_type::is_finite( x ) &&
           !traits_type::is_finite( traits_type::make( T( 0 ), T( CUDART_INF ) ) );
}

__global__ void check( bool *passed )
{
    *passed = check_real<float>() && check_real<double>() && check_complex<thrust::complex<float>>() &&
              check_complex<thrust::complex<double>>() && check_complex<cuFloatComplex>() &&
              check_complex<cuDoubleComplex>();
}
}

int main()
{
    static_assert( std::is_same<cufftComplex, cuFloatComplex>::value, "cuFFT must use the same specialization" );
    static_assert( std::is_same<cufftDoubleComplex, cuDoubleComplex>::value, "cuFFT must use the same specialization" );
    bool      *device     = nullptr;
    bool       passed     = false;
    const auto allocation = cudaMalloc( &device, sizeof( bool ) );
    if ( allocation != cudaSuccess )
    {
        std::fprintf( stderr, "CUDA scalar traits allocation failed: %s\n", cudaGetErrorString( allocation ) );
        return 1;
    }
    check<<<1, 1>>>( device );
    const auto launch  = cudaGetLastError();
    const auto copy    = cudaMemcpy( &passed, device, sizeof( bool ), cudaMemcpyDeviceToHost );
    const auto release = cudaFree( device );
    if ( launch != cudaSuccess || copy != cudaSuccess || release != cudaSuccess || !passed )
    {
        std::fprintf(
            stderr, "Device scalar traits failed: launch=%s copy=%s free=%s\n", cudaGetErrorString( launch ),
            cudaGetErrorString( copy ), cudaGetErrorString( release )
        );
        return 1;
    }
    std::puts( "CUDA scalar traits: real, Thrust and cuComplex checks passed" );
    return 0;
}
