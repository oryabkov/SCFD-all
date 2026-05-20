#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system_error.h>
#include <vector>

#include <scfd/backend/exclusive_scan/omp_exclusive_scan_impl.h>
#include <scfd/backend/exclusive_scan/thrust.h>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace
{

template <class T>
std::string value_to_string( const T &value )
{
    std::ostringstream out;
    out << value;
    return out.str();
}

template <class T>
std::vector<T> reference_exclusive_scan( const std::vector<T> &input, T init )
{
    std::vector<T> output( input.size() );
    T              sum = init;
    for ( std::size_t i = 0; i < input.size(); ++i )
    {
        output[i] = sum;
        sum += input[i];
    }
    return output;
}

template <class T>
void print_prefix( const std::vector<T> &values )
{
    const std::size_t print_size = std::min<std::size_t>( values.size(), 12 );
    std::cout << "[";
    for ( std::size_t i = 0; i < print_size; ++i )
    {
        if ( i != 0 )
        {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    if ( values.size() > print_size )
    {
        std::cout << ", ...";
    }
    std::cout << "]";
}

template <class T>
bool compare_vectors(
    const std::vector<T> &actual, const std::vector<T> &expected, const char *source, const std::string &case_name,
    T init, bool in_place, int threads
)
{
    if ( actual.size() != expected.size() )
    {
        std::cout << "FAILED " << source << " size mismatch in case " << case_name << ": actual " << actual.size()
                  << ", expected " << expected.size() << std::endl;
        return false;
    }

    for ( std::size_t i = 0; i < actual.size(); ++i )
    {
        if ( actual[i] != expected[i] )
        {
            std::cout << "FAILED " << source << " exclusive_scan in case " << case_name << ", size " << actual.size()
                      << ", init " << value_to_string( init ) << ", " << ( in_place ? "in-place" : "out-of-place" )
                      << ", omp_threads " << threads << ", first mismatch at index " << i << ": actual "
                      << value_to_string( actual[i] ) << ", expected " << value_to_string( expected[i] ) << std::endl;
            std::cout << "actual prefix   = ";
            print_prefix( actual );
            std::cout << std::endl << "expected prefix = ";
            print_prefix( expected );
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}

template <class T>
std::vector<T> run_omp_scan( const std::vector<T> &input, T init, bool in_place, int threads )
{
#ifdef _OPENMP
    omp_set_num_threads( threads );
#else
    (void)threads;
#endif
    scfd::omp_exclusive_scan<std::size_t> scan;

    if ( in_place )
    {
        std::vector<T> output = input;
        scan( output.size(), output.data(), output.data(), init );
        scan.wait();
        return output;
    }

    std::vector<T> output( input.size(), T( 0 ) );
    scan( input.size(), input.empty() ? nullptr : input.data(), output.empty() ? nullptr : output.data(), init );
    scan.wait();
    return output;
}

template <class T>
std::vector<T> run_cuda_scan( const std::vector<T> &input, T init, bool in_place )
{
    scfd::thrust_exclusive_scan<std::size_t> scan;

    thrust::device_vector<T> d_input( input.begin(), input.end() );
    if ( in_place )
    {
        T *ptr = d_input.empty() ? nullptr : thrust::raw_pointer_cast( d_input.data() );
        scan( d_input.size(), ptr, ptr, init );
        scan.wait();
        thrust::host_vector<T> host_output = d_input;
        return std::vector<T>( host_output.begin(), host_output.end() );
    }

    thrust::device_vector<T> d_output( input.size(), T( 0 ) );
    const T                 *input_ptr  = d_input.empty() ? nullptr : thrust::raw_pointer_cast( d_input.data() );
    T                       *output_ptr = d_output.empty() ? nullptr : thrust::raw_pointer_cast( d_output.data() );
    scan( d_input.size(), input_ptr, output_ptr, init );
    scan.wait();
    thrust::host_vector<T> host_output = d_output;
    return std::vector<T>( host_output.begin(), host_output.end() );
}

template <class T>
int run_case( const std::string &case_name, const std::vector<T> &input, T init, bool in_place, int threads )
{
    const std::vector<T> expected = reference_exclusive_scan( input, init );
    const std::vector<T> cuda_out = run_cuda_scan( input, init, in_place );
    const std::vector<T> omp_out  = run_omp_scan( input, init, in_place, threads );

    if ( !compare_vectors( cuda_out, expected, "CUDA/Thrust", case_name, init, in_place, threads ) )
    {
        return 1;
    }
    if ( !compare_vectors( omp_out, expected, "OpenMP", case_name, init, in_place, threads ) )
    {
        return 2;
    }
    if ( !compare_vectors( omp_out, cuda_out, "OpenMP vs CUDA/Thrust", case_name, init, in_place, threads ) )
    {
        return 3;
    }
    return 0;
}

std::vector<int> make_int_pattern( std::size_t size, int seed )
{
    std::vector<int> values( size );
    unsigned int     state = static_cast<unsigned int>( seed );
    for ( std::size_t i = 0; i < size; ++i )
    {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<int>( state % 23u ) - 11;
    }
    return values;
}

std::vector<long long> make_long_pattern( std::size_t size )
{
    std::vector<long long> values( size );
    for ( std::size_t i = 0; i < size; ++i )
    {
        values[i] = static_cast<long long>( i % 17 ) - 8;
    }
    return values;
}

std::vector<std::size_t> make_size_pattern( std::size_t size )
{
    std::vector<std::size_t> values( size );
    for ( std::size_t i = 0; i < size; ++i )
    {
        values[i] = ( i * 7 + 3 ) % 19;
    }
    return values;
}

template <class T>
int run_case_for_all_modes( const std::string &case_name, const std::vector<T> &input, T init )
{
    const int thread_counts[] = { 1, 2, 3, 4, 5, 7, 8, 16 };
    for ( int threads : thread_counts )
    {
        for ( int in_place = 0; in_place < 2; ++in_place )
        {
            const int result = run_case( case_name, input, init, in_place != 0, threads );
            if ( result != 0 )
            {
                return result;
            }
        }
    }
    return 0;
}

int run_all_cases()
{
    struct int_case
    {
        const char      *name;
        std::vector<int> input;
        int              init;
    };

    const int_case int_cases[] = {
        { "empty-int", std::vector<int>(), 0 },
        { "single-int-zero-init", std::vector<int>{ 42 }, 0 },
        { "single-int-negative-init", std::vector<int>{ 42 }, -11 },
        { "two-int", std::vector<int>{ 3, 4 }, 10 },
        { "small-positive-int", std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 }, 10 },
        { "small-mixed-int", std::vector<int>{ 3, -1, 0, 7, -4, 2, -9, 5, 1 }, -5 },
        { "static-boundary-31-int", make_int_pattern( 31, 17 ), 3 },
        { "static-boundary-32-int", make_int_pattern( 32, 19 ), -7 },
        { "static-boundary-33-int", make_int_pattern( 33, 23 ), 12 },
        { "static-boundary-64-int", make_int_pattern( 64, 29 ), -2 },
        { "static-boundary-65-int", make_int_pattern( 65, 31 ), 6 },
        { "irregular-1003-int", make_int_pattern( 1003, 37 ), -123 },
        { "large-8192-int", make_int_pattern( 8192, 41 ), 77 },
    };

    for ( const int_case &test_case : int_cases )
    {
        const int result = run_case_for_all_modes( test_case.name, test_case.input, test_case.init );
        if ( result != 0 )
        {
            return result;
        }
    }

    const int long_result =
        run_case_for_all_modes<long long>( "long-long-257", make_long_pattern( 257 ), -10000000000LL );
    if ( long_result != 0 )
    {
        return long_result;
    }

    const int size_result =
        run_case_for_all_modes<std::size_t>( "size-t-513", make_size_pattern( 513 ), std::size_t( 11 ) );
    if ( size_result != 0 )
    {
        return size_result;
    }

    return 0;
}

}

int main()
{
    try
    {
        int         device_count = 0;
        cudaError_t err          = cudaGetDeviceCount( &device_count );
        if ( err != cudaSuccess || device_count <= 0 )
        {
            std::cout << "FAILED cudaGetDeviceCount: " << cudaGetErrorString( err ) << std::endl;
            return 10;
        }

        const int result = run_all_cases();
        if ( result != 0 )
        {
            return 20 + result;
        }
    }
    catch ( const thrust::system_error &err )
    {
        std::cout << "FAILED thrust system_error: " << err.what() << std::endl;
        return 30;
    }
    catch ( const std::exception &err )
    {
        std::cout << "FAILED exception: " << err.what() << std::endl;
        return 31;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
