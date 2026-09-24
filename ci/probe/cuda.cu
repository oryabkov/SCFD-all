#include <cuda_runtime.h>

#include <iostream>

__global__ void write_value( int *value )
{
    *value = 42;
}

int main()
{
    int        *device = nullptr;
    int         result = 0;
    cudaError_t status = cudaMalloc( reinterpret_cast<void **>( &device ), sizeof( int ) );
    if ( status == cudaSuccess )
    {
        write_value<<<1, 1>>>( device );
        status = cudaGetLastError();
    }
    if ( status == cudaSuccess )
        status = cudaMemcpy( &result, device, sizeof( int ), cudaMemcpyDeviceToHost );
    if ( device != nullptr )
    {
        const cudaError_t free_status = cudaFree( device );
        if ( status == cudaSuccess )
            status = free_status;
    }
    if ( status != cudaSuccess || result != 42 )
    {
        std::cerr << "CUDA capability check failed: " << cudaGetErrorString( status ) << " (value " << result << ")\n";
        return 1;
    }
    std::cout << "CUDA device allocation, kernel and transfer succeeded\n";
    return 0;
}
