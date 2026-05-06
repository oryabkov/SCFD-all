// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Fountion, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTIHIPLAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __HIPBLAS_SAFE_CALL_H__
#define __HIPBLAS_SAFE_CALL_H__

#include <hipblas/hipblas.h>
#include <sstream>

#define __STR_HELPER( x ) #x
#define __STR( x ) __STR_HELPER( x )

static const char *_hipblasGetErrorEnum( hipblasStatus_t error )
{
    switch ( error )
    {
    case HIPBLAS_STATUS_SUCCESS:
        return "The operation completed successfully.";

    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "The hipBLAS library was not initialized. This is usually caused by the lack of a prior hipblasCreate() "
               "call, an error in the HIP Runtime API called by the hipBLAS routine, or an error in the hardware "
               "setup.";

    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the hipBLAS library. This is usually caused by a hipMalloc() "
               "failure.";

    case HIPBLAS_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";

    case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of "
               "support for double precision.";

    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to exehipte. This is often caused by a launch failure of the kernel on the GPU, "
               "which can be caused by multiple reasons.";

    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "An internal hipBLAS operation failed. This error is usually caused by a hipMemcpyAsync() failure.";

    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "The functionnality requested is not supported.";

    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
        return "The hipBLAS handle is null pointer";

    case HIPBLAS_STATUS_INVALID_ENUM:
        return "unsupported enum value was passed to function";

    case HIPBLAS_STATUS_UNKNOWN:
        return "back-end returned an unsupported status code";
    }

    return "<unknown>";
}

#define HIPBLAS_SAFE_CALL( X )                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        hipblasStatus_t status  = ( X );                                                                               \
        hipError_t      hip_res = hipDeviceSynchronize();                                                              \
        if ( status != HIPBLAS_STATUS_SUCCESS )                                                                        \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << std::string( "HIPBLAS_SAFE_CALL " __FILE__ " " __STR( __LINE__ ) " : " #X " failed: " )              \
               << std::string( _hipblasGetErrorEnum( status ) );                                                       \
            std::string str = ss.str();                                                                                \
            throw std::runtime_error( str );                                                                           \
        }                                                                                                              \
        if ( hip_res != hipSuccess )                                                                                   \
            throw std::runtime_error(                                                                                  \
                std::string(                                                                                           \
                    "HIPBLAS_SAFE_CALL " __FILE__ " " __STR( __LINE__ ) " : " #X " failed hipDeviceSynchronize: "      \
                ) +                                                                                                    \
                std::string( hipGetErrorString( hip_res ) )                                                            \
            );                                                                                                         \
    } while ( 0 )


#endif
