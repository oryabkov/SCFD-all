#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <scfd/arrays/array.h>
#include <scfd/memory/hip.h>
#include <scfd/external_libraries/hipblas_wrap_singleton_impl.h>

namespace
{
int checks   = 0;
int failures = 0;

template <class T>
std::complex<double> host_value( const T &value )
{
    if constexpr ( std::is_floating_point<T>::value )
        return { double( value ), 0 };
    else if constexpr ( std::is_same<T, hipComplex>::value || std::is_same<T, hipDoubleComplex>::value )
        return { double( value.x ), double( value.y ) };
    else
        return { double( value.real() ), double( value.imag() ) };
}

template <class T>
T make_value( double real, double imag = 0 )
{
    using real_type = typename scfd::hipblas_real_types::hipblas_real_type_hlp<T>::type;
    if constexpr ( std::is_floating_point<T>::value )
        return T( real );
    else if constexpr ( std::is_same<T, hipComplex>::value || std::is_same<T, hipDoubleComplex>::value )
        return { real_type( real ), real_type( imag ) };
    else
        return T( real_type( real ), real_type( imag ) );
}

template <class T>
void require_close( const T &actual, std::complex<double> expected, const char *message )
{
    using real_type        = typename scfd::hipblas_real_types::hipblas_real_type_hlp<T>::type;
    const double tolerance = 100 * std::numeric_limits<real_type>::epsilon() * ( 1 + std::abs( expected ) );
    ++checks;
    if ( !( std::abs( host_value( actual ) - expected ) <= tolerance ) )
    {
        ++failures;
        std::cerr << "FAIL " << message << ": " << host_value( actual ) << " != " << expected << '\n';
    }
}

template <class T>
void test_blas( scfd::hipblas_wrap &blas, const char *name )
{
    using real_type   = typename scfd::hipblas_real_types::hipblas_real_type_hlp<T>::type;
    using vector_type = scfd::arrays::array<T, scfd::memory::hip_device>;
    vector_type x, y, z;
    x.init( 4 );
    y.init( 4 );
    z.init( 4 );
    const T a[4] = { make_value<T>( 1, 2 ), make_value<T>( -3, 1 ), make_value<T>( 2, -1 ), make_value<T>( 4, 3 ) };
    const T b[4] = { make_value<T>( 2, -1 ), make_value<T>( 1, 3 ), make_value<T>( -2, 2 ), make_value<T>( 3, -2 ) };
    blas.set_vector( 4, a, x.raw_ptr() );
    blas.set_vector( 4, b, y.raw_ptr() );
    std::complex<double> dot     = 0;
    double               norm_sq = 0, asum = 0;
    for ( int i = 0; i < 4; ++i )
    {
        dot += std::conj( host_value( a[i] ) ) * host_value( b[i] );
        norm_sq += std::norm( host_value( a[i] ) );
        asum += std::abs( host_value( a[i] ).real() ) + std::abs( host_value( a[i] ).imag() );
    }
    T scalar;
    blas.dot( 4, x.raw_ptr(), y.raw_ptr(), &scalar );
    require_close( scalar, dot, "dot conjugates first argument" );
    real_type result;
    blas.norm2( 4, x.raw_ptr(), &result );
    require_close( result, std::sqrt( norm_sq ), "norm2" );
    blas.asum( 4, x.raw_ptr(), &result );
    require_close( result, asum, "asum" );
    blas.set_pointer_location_device( true );
    blas.dot( 4, x.raw_ptr(), y.raw_ptr(), z.raw_ptr() );
    blas.get_vector( 1, z.raw_ptr(), &scalar );
    blas.set_pointer_location_device( false );
    require_close( scalar, dot, "device scalar result" );

    const T alpha = make_value<T>( 2, -1 );
    blas.copy( 4, x.raw_ptr(), z.raw_ptr() );
    blas.scale( 4, alpha, z.raw_ptr() );
    blas.axpy( 4, alpha, x.raw_ptr(), y.raw_ptr() );
    T values[4];
    blas.get_vector( 4, z.raw_ptr(), values );
    for ( int i = 0; i < 4; ++i )
        require_close( values[i], host_value( alpha ) * host_value( a[i] ), "copy and scale" );
    blas.get_vector( 4, y.raw_ptr(), values );
    for ( int i = 0; i < 4; ++i )
        require_close( values[i], host_value( b[i] ) + host_value( alpha ) * host_value( a[i] ), "axpy" );
    if constexpr ( !std::is_floating_point<T>::value )
    {
        blas.template scale<T>( 4, real_type( 0.5 ), z.raw_ptr() );
        blas.get_vector( 4, z.raw_ptr(), values );
        for ( int i = 0; i < 4; ++i )
            require_close( values[i], 0.5 * host_value( alpha ) * host_value( a[i] ), "real complex scaling" );
    }

    // The wrapper exposes matrix operations for real and thrust complex scalars.
    if constexpr ( std::is_floating_point<T>::value || std::is_same<T, thrust::complex<real_type>>::value )
    {
        blas.set_matrix( 2, 2, a, 2, x.raw_ptr(), 2 );
        blas.set_vector( 4, b, y.raw_ptr() );
        const T zero = make_value<T>( 0 );
        blas.gemv( 'N', 2, x.raw_ptr(), 2, 2, alpha, y.raw_ptr(), zero, z.raw_ptr() );
        blas.get_vector( 2, z.raw_ptr(), values );
        for ( int i = 0; i < 2; ++i )
            require_close(
                values[i],
                host_value( alpha ) *
                    ( host_value( a[i] ) * host_value( b[0] ) + host_value( a[i + 2] ) * host_value( b[1] ) ),
                "gemv"
            );
        blas.gemm( 'N', 'N', 2, 2, 2, alpha, x.raw_ptr(), 2, y.raw_ptr(), 2, zero, z.raw_ptr(), 2 );
        blas.get_matrix( 2, 2, z.raw_ptr(), 2, values, 2 );
        for ( int j = 0; j < 2; ++j )
            for ( int i = 0; i < 2; ++i )
                require_close(
                    values[i + 2 * j],
                    host_value( alpha ) * ( host_value( a[i] ) * host_value( b[2 * j] ) +
                                            host_value( a[i + 2] ) * host_value( b[1 + 2 * j] ) ),
                    "gemm"
                );
    }
    std::cout << "Checked " << name << '\n';
}
}

int main()
{
    try
    {
        scfd::hipblas_wrap blas;
        test_blas<float>( blas, "float" );
        test_blas<double>( blas, "double" );
        test_blas<scfd::hipblas_compat::complex_float>( blas, "native complex float" );
        test_blas<scfd::hipblas_compat::complex_double>( blas, "native complex double" );
        test_blas<thrust::complex<float>>( blas, "thrust complex float" );
        test_blas<thrust::complex<double>>( blas, "thrust complex double" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "hipBLAS " << hipblasVersionMajor << '.' << hipblasVersionMinor << ": " << checks << " checks, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
