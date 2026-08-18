// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTIHIPLAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_UTILS_HIPSOLVER_SAFE_CALL_H__
#define __SCFD_UTILS_HIPSOLVER_SAFE_CALL_H__

#include <hipsolver/hipsolver.h>
#include <sstream>
#include <stdexcept>

#define __STR_HELPER( x ) #x
#define __STR( x ) __STR_HELPER( x )

static const char *_hipsolverGetErrorEnum( hipsolverStatus_t error )
{
    switch ( error )
    {
    case HIPSOLVER_STATUS_SUCCESS:
        return "The operation completed successfully.";

    case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "the matrix type is not supported.";

    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        return "The hipsolver library was not initialized. This is usually caused by the lack of a prior "
               "hipsolverCreate() call, an error in the HIP Runtime API called by the hipsolver routine, or an error "
               "in "
               "the hardware setup.";

    case HIPSOLVER_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the hipsolver library. This is usually caused by a hipMalloc() "
               "failure.";

    case HIPSOLVER_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";

    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of "
               "support for double precision.";

    case HIPSOLVER_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to exehipte. This is often caused by a launch failure of the kernel on the GPU, "
               "which can be caused by multiple reasons.";

    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        return "An internal hipBLAS operation failed. This error is usually caused by a hipMemcpyAsync() failure.";

    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        return "The functionnality requested is not supported.";

    case HIPSOLVER_STATUS_ZERO_PIVOT:
        return "a zero pivot was encountered during the computation.";

    case HIPSOLVER_STATUS_UNKNOWN:
        return "back-end returned an unsupported status code";

    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        return "hipSOLVER handle is null pointer";
    case HIPSOLVER_STATUS_INVALID_ENUM:
        return "unsupported enum value was passed to function";
    }

    return "<unknown>";
}

#define HIPSOLVER_SAFE_CALL( X )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        hipsolverStatus_t status  = ( X );                                                                             \
        hipError_t        hip_res = hipDeviceSynchronize();                                                            \
        if ( status != HIPSOLVER_STATUS_SUCCESS )                                                                      \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << std::string( "HIPSOLVER_SAFE_CALL " __FILE__ " " __STR( __LINE__ ) " : " #X " failed: " )            \
               << std::string( _hipsolverGetErrorEnum( status ) );                                                     \
            std::string str = ss.str();                                                                                \
            throw std::runtime_error( str );                                                                           \
        }                                                                                                              \
        if ( hip_res != hipSuccess )                                                                                   \
            throw std::runtime_error(                                                                                  \
                std::string(                                                                                           \
                    "HIPSOLVER_SAFE_CALL " __FILE__ " " __STR( __LINE__ ) " : " #X " failed hipDeviceSynchronize: "    \
                ) +                                                                                                    \
                std::string( hipGetErrorString( hip_res ) )                                                            \
            );                                                                                                         \
    } while ( 0 )


#endif
