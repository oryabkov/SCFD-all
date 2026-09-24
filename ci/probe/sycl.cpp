#include <sycl/sycl.hpp>

// Existing SCFD SYCL tests also require these oneDPL headers.
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <iostream>

int main()
{
    try
    {
        sycl::queue queue( sycl::gpu_selector_v );
        int        *device = sycl::malloc_device<int>( 1, queue );
        if ( device == nullptr )
            return 1;
        int result = 0;
        try
        {
            queue.single_task( [=]() { *device = 42; } ).wait_and_throw();
            queue.memcpy( &result, device, sizeof( int ) ).wait_and_throw();
        }
        catch ( ... )
        {
            sycl::free( device, queue );
            throw;
        }
        sycl::free( device, queue );
        if ( result != 42 )
            return 1;
        std::cout << "SYCL GPU allocation, kernel and transfer succeeded\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "SYCL capability check failed: " << error.what() << '\n';
        return 1;
    }
}
