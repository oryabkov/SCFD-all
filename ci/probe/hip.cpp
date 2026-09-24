#include <hip/hip_runtime.h>

// hipcc supplies the appropriate HIP/Thrust backend for AMD or NVIDIA.
// Do not include SCFD headers: library regressions belong to the main build.
#include <hipblas/hipblas.h>
#include <hipblas/hipblas-version.h>
#include <hipsolver/hipsolver.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <iostream>

__global__ void write_value( int *value )
{
    *value = 42;
}

int main()
{
    int       *device = nullptr;
    int        result = 0;
    hipError_t status = hipMalloc( reinterpret_cast<void **>( &device ), sizeof( int ) );
    if ( status == hipSuccess )
    {
        hipLaunchKernelGGL( write_value, dim3( 1 ), dim3( 1 ), 0, 0, device );
        status = hipGetLastError();
    }
    if ( status == hipSuccess )
        status = hipMemcpy( &result, device, sizeof( int ), hipMemcpyDeviceToHost );
    if ( device != nullptr )
    {
        const hipError_t free_status = hipFree( device );
        if ( status == hipSuccess )
            status = free_status;
    }
    if ( status != hipSuccess || result != 42 )
    {
        std::cerr << "HIP capability check failed: " << hipGetErrorString( status ) << " (value " << result << ")\n";
        return 1;
    }
    std::cout << "HIP device allocation, kernel and transfer succeeded\n";
    return 0;
}
