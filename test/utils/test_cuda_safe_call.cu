// Copyright © 2026 SCFD contributors
// SPDX-License-Identifier: GPL-2.0-only

#include <iostream>
#include <stdexcept>
#include <string>

#include <scfd/utils/cuda_safe_call.h>

int main()
{
    SCFD_CUDA_SAFE_CALL( cudaSetDevice( 0 ) );
    int *device = nullptr;
    int  input = 193, output = 0;
    SCFD_CUDA_SAFE_CALL( cudaMalloc( reinterpret_cast<void **>( &device ), sizeof( int ) ) );
    SCFD_CUDA_SAFE_CALL( cudaMemcpy( device, &input, sizeof( int ), cudaMemcpyHostToDevice ) );
    SCFD_CUDA_SAFE_CALL( cudaMemcpy( &output, device, sizeof( int ), cudaMemcpyDeviceToHost ) );
    SCFD_CUDA_SAFE_CALL( cudaFree( device ) );
    if ( output != input )
        return 1;

    bool caught = false;
    try
    {
        // Exercise error propagation without exhausting memory or damaging the context.
        SCFD_CUDA_SAFE_CALL( cudaErrorInvalidValue );
    }
    catch ( const std::runtime_error &error )
    {
        const std::string message( error.what() );
        caught = message.find( "cudaErrorInvalidValue failed" ) != std::string::npos &&
                 message.find( cudaGetErrorString( cudaErrorInvalidValue ) ) != std::string::npos;
    }
    if ( !caught )
        return 2;
    std::cout << "CUDA allocation/copy and expected safe-call error passed" << std::endl;
    return 0;
}
