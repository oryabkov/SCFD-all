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

#ifndef __SCFD_HIPBLAS_WRAP_H__
#define __SCFD_HIPBLAS_WRAP_H__

#if !defined( __HIPCC__ )
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM 5
#endif
#endif

#include <hipblas/hipblas.h>
#include <hipblas/hipblas-version.h>
#include <thrust/complex.h>
#include <scfd/utils/hipblas_safe_call.h>
#include <scfd/utils/manual_init_singleton.h>
#include <stdexcept>
#include <iostream>

namespace scfd
{

namespace hipblas_complex_types
{
template <typename T>
struct hipblas_hipComplex_type_hlp
{
};


template <>
struct hipblas_hipComplex_type_hlp<float>
{
    typedef hipblasComplex type;
};

template <>
struct hipblas_hipComplex_type_hlp<double>
{
    typedef hipblasDoubleComplex type;
};
}

namespace hipblas_real_types
{
template <typename T>
struct hipblas_real_type_hlp
{
};

template <>
struct hipblas_real_type_hlp<float>
{
    typedef float type;
};

template <>
struct hipblas_real_type_hlp<double>
{
    typedef double type;
};

template <>
struct hipblas_real_type_hlp<hipblasComplex>
{
    typedef float type;
};

template <>
struct hipblas_real_type_hlp<hipblasDoubleComplex>
{
    typedef double type;
};

template <>
struct hipblas_real_type_hlp<thrust::complex<float>>
{
    typedef float type;
};

template <>
struct hipblas_real_type_hlp<thrust::complex<double>>
{
    typedef double type;
};
}

class hipblas_wrap : public utils::manual_init_singleton<hipblas_wrap>
{
public:
    hipblas_wrap( bool do_set_inst = false ) : manual_init_singleton( this, do_set_inst ), handle_created( false )
    {
        hipblas_create();
        handle_created = true;
        set_pointer_location_device( false );
    }

    hipblas_wrap( bool plot_info, bool do_set_inst )
        : manual_init_singleton( this, do_set_inst ), handle_created( false )
    {
        if ( plot_info )
        {
            hipblas_create_info();
            handle_created = true;
        }
        else
        {
            hipblas_create();
            handle_created = true;
        }
        set_pointer_location_device( false );
    }

    hipblas_wrap( const hipblas_wrap & ) = delete;
    hipblas_wrap( hipblas_wrap &&w )
    {
        operator=( std::move( w ) );
    }

    hipblas_wrap &operator=( const hipblas_wrap & ) = delete;
    hipblas_wrap &operator=( hipblas_wrap &&w )
    {
        handle_created   = w.handle_created;
        handle           = w.handle;
        w.handle_created = false;

        return *this;
    }


    ~hipblas_wrap()
    {
        if ( handle_created )
        {
            hipblas_destroy();
            handle_created = false;
        }
    }

    hipblasHandle_t *get_handle()
    {
        return &handle;
    }


    void set_stream( hipStream_t streamId )
    {
        HIPBLAS_SAFE_CALL( hipblasSetStream( handle, streamId ) );
    }

    hipStream_t get_stream()
    {
        hipStream_t streamId;
        HIPBLAS_SAFE_CALL( hipblasGetStream( handle, &streamId ) );
        return streamId;
    }

    //where scalar results are stored, like dot product etc.
    hipblasPointerMode_t get_pointer_location()
    {
        hipblasPointerMode_t mode;
        HIPBLAS_SAFE_CALL( hipblasGetPointerMode( handle, &mode ) );
        return mode;
    }

    //where to store scalar results, like dot product etc.
    // true means store on device
    //this scalar pointer must be allocated on the device via hipMalloc!
    void set_pointer_location_device( bool store_on_device )
    {
        if ( store_on_device )
        {
            HIPBLAS_SAFE_CALL( hipblasSetPointerMode( handle, HIPBLAS_POINTER_MODE_DEVICE ) );
            scalar_pointer_on_device = true;
        }
        else
        {
            HIPBLAS_SAFE_CALL( hipblasSetPointerMode( handle, HIPBLAS_POINTER_MODE_HOST ) );
            scalar_pointer_on_device = false;
        }
    }

    template <typename T>
    void set_vector( size_t vector_size, const T *vec_host, T *vec_device, int incx = 1, int incy = 1 )
    {
        HIPBLAS_SAFE_CALL( hipblasSetVector( vector_size, sizeof( T ), vec_host, incx, vec_device, incy ) );
    }
    template <typename T>
    void get_vector( size_t vector_size, const T *vec_device, T *vec_host, int incx = 1, int incy = 1 )
    {
        HIPBLAS_SAFE_CALL( hipblasGetVector( vector_size, sizeof( T ), vec_device, incx, vec_host, incy ) );
    }
    template <typename T>
    void set_matrix( size_t rows, size_t cols, const T *mat_host, int lda, T *mat_device, int ldb )
    {
        HIPBLAS_SAFE_CALL( hipblasSetMatrix( rows, cols, sizeof( T ), mat_host, lda, mat_device, ldb ) );
    }
    template <typename T>
    void get_matrix( size_t rows, size_t cols, const T *mat_device, int lda, T *mat_host, int ldb )
    {
        HIPBLAS_SAFE_CALL( hipblasGetMatrix( rows, cols, sizeof( T ), mat_device, lda, mat_host, ldb ) );
    }
    template <typename T>
    void set_vector_async(
        size_t vector_size, const T *vec_host, T *vec_device, hipStream_t stream, int incx = 1, int incy = 1
    )
    {
        HIPBLAS_SAFE_CALL( hipblasSetVectorAsync( vector_size, sizeof( T ), vec_host, incx, vec_device, incy, stream ) );
    }
    template <typename T>
    void get_vector_async(
        size_t vector_size, const T *vec_device, T *vec_host, hipStream_t stream, int incx = 1, int incy = 1
    )
    {
        HIPBLAS_SAFE_CALL( hipblasGetVectorAsync( vector_size, sizeof( T ), vec_device, incx, vec_host, incy, stream ) );
    }
    template <typename T>
    void set_matrix_async(
        size_t rows, size_t cols, const T *mat_host, T *mat_device, hipStream_t stream, int lda = 1, int ldb = 1
    )
    {
        HIPBLAS_SAFE_CALL( hipblasSetMatrixAsync( rows, cols, sizeof( T ), mat_host, lda, mat_device, ldb, stream ) );
    }
    template <typename T>
    void get_matrix_async(
        size_t rows, size_t cols, const T *mat_device, T *mat_host, hipStream_t stream, int lda = 1, int ldb = 1
    )
    {
        HIPBLAS_SAFE_CALL( hipblasGetMatrixAsync( rows, cols, sizeof( T ), mat_device, lda, mat_host, ldb, stream ) );

    }
    
    void use_tensor_core_operations(bool useTCO)
    {
        hipblasMath_t mode = HIPBLAS_DEFAULT_MATH;
        if(useTCO)
            mode = HIPBLAS_TENSOR_OP_MATH;

        HIPBLAS_SAFE_CALL(hipblasSetMathMode(handle, mode));

    }

    //===hipBLAS Level-1 Functions=== see: https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/hipblas-api-functions.html#level-1-blas
    template <typename T>
    void sum_abs_elements(
        size_t vector_size, const T *vector, typename hipblas_real_types::hipblas_real_type_hlp<T>::type *result,
        int incx = 1
    );

    // y [ j ] = alpha x [ k ] + y [ j ]
    template <typename T>
    void axpy( size_t vector_sizes, const T alpha, const T *x, T *y, int incx = 1, int incy = 1 );
    // y [ j ] = x [ k ]
    template <typename T>
    void copy( size_t vector_sizes, const T *x, T *y, int incx = 1, int incy = 1 );
    // y [ j ] <-> x [ k ]
    template <typename T>
    void swap( size_t vector_sizes, T *x, T *y, int incx = 1, int incy = 1 );
    //dot product (we use automatic conjugation for complex number, i.e. dot(u,v)=u^C*v)
    template <typename T>
    void dot( size_t vector_size, const T *x, const T *y, T *result, int incx = 1, int incy = 1 );
    //sbsolute sum of a vector
    template <typename T>
    void asum(
        size_t vector_size, const T *x, typename hipblas_real_types::hipblas_real_type_hlp<T>::type *result, int incx = 1
    );
    //vector l2 norm.
    template <typename T>
    void norm2(
        size_t vector_size, const T *x, typename hipblas_real_types::hipblas_real_type_hlp<T>::type *result, int incx = 1
    );
    //scale vector as x=x*a. 'a' can be real or complex
    template <typename T>
    void scale( size_t vector_size, const T alpha, T *x, int incx = 1 );
    template <typename T>
    void scale(
        size_t vector_size, const typename hipblas_real_types::hipblas_real_type_hlp<T>::type alpha, T *x, int incx = 1
    );
    //normalizes vector. Overwrites a vector with normalized one and returns it's norm. Usually used in Krylov-type methods
    template <typename T>
    void normalize( size_t vector_size, T *x, T *norm, int incx = 1 );
    template <typename T>
    void normalize(
        size_t vector_size, T *x, typename hipblas_real_types::hipblas_real_type_hlp<T>::type *norm, int incx = 1
    );
    //TODO: add Givens rotations construction for Arnoldi process

    //===hipBLAS Level-2 Functions=== see: https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/hipblas-api-functions.html#level-2-blas
private:
    hipblasOperation_t switch_operation_real( const char &op )
    {
        hipblasOperation_t operation = HIPBLAS_OP_N;
        switch ( op )
        {
        case 'N':
            operation = HIPBLAS_OP_N;
            break;
        case 'T':
            operation = HIPBLAS_OP_T;
            break;
        default:
            // invalid operation code throw
            throw std::runtime_error(
                "switch_operation_real: invalid code for original or transpose operations. Only 'N' or 'T' are defined."
            );
        }
        return operation;
    }

    hipblasOperation_t switch_operation_complex( const char &op )
    {
        hipblasOperation_t operation = HIPBLAS_OP_N;
        switch ( op )
        {
        case 'N':
            operation = HIPBLAS_OP_N;
            break;
        case 'T':
            operation = HIPBLAS_OP_C; // HIPBLAS_OP_H is defined in documentaiton?!?
                                     //definition in:
                                     // ../hip/include/hipblas_api.h
            break;
        default:
            // invalid operation code throw
            throw std::runtime_error( "switch_operation_complex: invalid code for original or transpose operations. "
                                      "Only 'N' or 'T' (for Hermitian transpose) are defined." );
        }
        return operation;
    }

public:
    //This function performs the matrix-vector multiplication:
    //                  y = α op ( A ) x + β y,
    //where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
    //Also, for matrix A:
    //   op(A) = A  if transa == HIPBLAS_OP_N
    //   op(A) = A^T  if transa == HIPBLAS_OP_T
    //   op(A) = A^H  if transa == HIPBLAS_OP_H
    //   op = 'N' for HIPBLAS_OP_N, op = 'T' for HIPBLAS_OP_T, op = 'H' for HIPBLAS_OP_H
    //LDimA is the leading dimension of A, for C arrays LDimA = RowA.
    template <typename T>
    void gemv( const char op, size_t RowA, const T *A, size_t ColA, size_t LDimA, T alpha, const T *x, T beta, T *y );


    //===hipBLAS Level-3 Functions=== see: https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/hipblas-api-functions.html#level-3-blas
    // hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
    //                            hipblasOperation_t transa, hipblasOperation_t transb,
    //                            int m, int n, int k,
    //                            const float           *alpha,
    //                            const float           *A, int lda,
    //                            const float           *B, int ldb,
    //                            const float           *beta,
    //                            float           *C, int ldc)
    template <typename T>
    void gemm(
        char opA, char opB, size_t RowAC, size_t ColBC, size_t ColARowB, T alpha, const T *A, size_t LDimA, const T *B,
        size_t LDimB, T beta, T *C, size_t LdimC
    );

    //===hipBLAS BLAS-like EXTENSIONS===

    // This function solves the triangular linear system with multiple right-hand-sides
    // op ( A ) X = α B if  side == HIPBLAS_SIDE_LEFT X op ( A ) = α B if  side == HIPBLAS_SIDE_RIGHT
    // where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m × n matrices, and α is a scalar. Also, for matrix A
    // op ( A ) = A if  transa == HIPBLAS_OP_N A T if  transa == HIPBLAS_OP_T A H if  transa == HIPBLAS_OP_C
    // The solution X overwrites the right-hand-sides B on exit.
    // No test for singularity or near-singularity is included in this function.

    // m: number of rows of matrix B, with matrix A sized accordingly.
    // n: number of columns of matrix B, with matrix A is sized accordingly.
    // A: device, input, <type> array of dimension lda x m with lda>=max(1,m) if side == HIPBLAS_SIDE_LEFT and lda x n with lda>=max(1,n) otherwise.
    // B: device, in/out, <type> array. It has dimensions ldb x n with ldb>=max(1,m).

    template <typename T>
    void trsm(
        const char sideA, const char isA_UorL, const char opA, bool unit_diag_A, size_t RowBColA, size_t ColsB,
        const T alpha, const T *A, size_t LDimA, T *B, size_t LDimB
    );


    template <typename T>
    void geam(
        const char opA, size_t RowAC, size_t ColBC, const T alpha, const T *A, size_t LDimA, const T beta, T *B,
        size_t LDimB, T *C, size_t LDimC
    );

private:
    hipblasHandle_t handle;
    bool           handle_created;
    bool           scalar_pointer_on_device;


    void hipblas_create()
    {

        HIPBLAS_SAFE_CALL( hipblasCreate( &handle ) );
    }

    void hipblas_destroy()
    {

        HIPBLAS_SAFE_CALL( hipblasDestroy( handle ) );
    }

    void hipblas_create_info()
    {
        HIPBLAS_SAFE_CALL( hipblasCreate( &handle ) );
        const int major_ver   = hipblasVersionMajor;
        const int minor_ver   = hipblasVersionMinor;
        const int patch_level = hipblasVersionPatch;
        const int hipblas_version = major_ver * 1000 + minor_ver * 10 + patch_level;

        std::cout << "hipBLAS v." << hipblas_version << " (major=" << major_ver << ", minor=" << minor_ver
                  << ", patch level=" << patch_level << ") handle created." << std::endl;
    }
};

// template specializations for level 1 BLAS functions

template <>
inline void hipblas_wrap::sum_abs_elements( size_t vector_size, const float *vector, float *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasSasum( handle, vector_size, vector, incx, result ) );
}
template <>
inline void hipblas_wrap::sum_abs_elements( size_t vector_size, const double *vector, double *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasDasum( handle, vector_size, vector, incx, result ) );
}
template <>
inline void hipblas_wrap::sum_abs_elements(
    size_t vector_size, const thrust::complex<float> *vector,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<float>>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScasum( handle, vector_size, (hipblasComplex *)vector, incx, result ) );
}
template <>
inline void hipblas_wrap::sum_abs_elements(
    size_t vector_size, const thrust::complex<double> *vector,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<double>>::type *result, int incx
)
{

    HIPBLAS_SAFE_CALL( hipblasDzasum( handle, vector_size, (hipblasDoubleComplex *)vector, incx, result ) );
}
template <>
inline void hipblas_wrap::sum_abs_elements(
    size_t vector_size, const hipblasComplex *vector,
    typename hipblas_real_types::hipblas_real_type_hlp<hipblasComplex>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScasum( handle, vector_size, vector, incx, result ) );
}
template <>
inline void hipblas_wrap::sum_abs_elements(
    size_t vector_size, const hipblasDoubleComplex *vector,
    typename hipblas_real_types::hipblas_real_type_hlp<hipblasDoubleComplex>::type *result, int incx
)
{

    HIPBLAS_SAFE_CALL( hipblasDzasum( handle, vector_size, vector, incx, result ) );
}
//This function multiplies the vector x by the scalar alpha
//and adds it to the vector y overwriting the latest vector with the result.
//y=alpha*x+y;
template <>
inline void hipblas_wrap::axpy( size_t vector_sizes, const float alpha, const float *x, float *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasSaxpy( handle, vector_sizes, &alpha, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::axpy( size_t vector_sizes, const double alpha, const double *x, double *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasDaxpy( handle, vector_sizes, &alpha, x, incx, y, incy ) );
}
template <>
inline void
hipblas_wrap::axpy( size_t vector_sizes, const hipblasComplex alpha, const hipblasComplex *x, hipblasComplex *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCaxpy( handle, vector_sizes, &alpha, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::axpy(
    size_t vector_sizes, const hipblasDoubleComplex alpha, const hipblasDoubleComplex *x, hipblasDoubleComplex *y, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL( hipblasZaxpy( handle, vector_sizes, &alpha, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::axpy(
    size_t vector_sizes, const thrust::complex<float> alpha, const thrust::complex<float> *x, thrust::complex<float> *y,
    int incx, int incy
)
{
    HIPBLAS_SAFE_CALL(
        hipblasCaxpy( handle, vector_sizes, (hipblasComplex *)&alpha, (hipblasComplex *)x, incx, (hipblasComplex *)y, incy )
    );
}
template <>
inline void hipblas_wrap::axpy(
    size_t vector_sizes, const thrust::complex<double> alpha, const thrust::complex<double> *x,
    thrust::complex<double> *y, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL( hipblasZaxpy(
        handle, vector_sizes, (hipblasDoubleComplex *)&alpha, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)y, incy
    ) );
}
//
template <>
inline void hipblas_wrap::copy( size_t vector_sizes, const float *x, float *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasScopy( handle, vector_sizes, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::copy( size_t vector_sizes, const double *x, double *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasDcopy( handle, vector_sizes, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::copy( size_t vector_sizes, const hipblasComplex *x, hipblasComplex *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCcopy( handle, vector_sizes, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::copy( size_t vector_sizes, const hipblasDoubleComplex *x, hipblasDoubleComplex *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasZcopy( handle, vector_sizes, x, incx, y, incy ) );
}
template <>
inline void
hipblas_wrap::copy( size_t vector_sizes, const thrust::complex<float> *x, thrust::complex<float> *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCcopy( handle, vector_sizes, (hipblasComplex *)x, incx, (hipblasComplex *)y, incy ) );
}
template <>
inline void hipblas_wrap::copy(
    size_t vector_sizes, const thrust::complex<double> *x, thrust::complex<double> *y, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL( hipblasZcopy( handle, vector_sizes, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)y, incy ) );
}
//
template <>
inline void hipblas_wrap::swap( size_t vector_size, float *x, float *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasSswap( handle, vector_size, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::swap( size_t vector_size, double *x, double *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasDswap( handle, vector_size, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::swap( size_t vector_size, hipblasComplex *x, hipblasComplex *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCswap( handle, vector_size, x, incx, y, incy ) );
}
template <>
inline void hipblas_wrap::swap( size_t vector_size, hipblasDoubleComplex *x, hipblasDoubleComplex *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasZswap( handle, vector_size, x, incx, y, incy ) );
}
template <>
inline void
hipblas_wrap::swap( size_t vector_size, thrust::complex<float> *x, thrust::complex<float> *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCswap( handle, vector_size, (hipblasComplex *)x, incx, (hipblasComplex *)y, incy ) );
}
template <>
inline void
hipblas_wrap::swap( size_t vector_size, thrust::complex<double> *x, thrust::complex<double> *y, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasZswap( handle, vector_size, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)y, incy ) );
}
//
template <>
inline void hipblas_wrap::dot( size_t vector_size, const float *x, const float *y, float *result, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasSdot( handle, vector_size, x, incx, y, incy, result ) );
}
template <>
inline void hipblas_wrap::dot( size_t vector_size, const double *x, const double *y, double *result, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasDdot( handle, vector_size, x, incx, y, incy, result ) );
}
template <>
inline void
hipblas_wrap::dot( size_t vector_size, const hipblasComplex *x, const hipblasComplex *y, hipblasComplex *result, int incx, int incy )
{
    HIPBLAS_SAFE_CALL( hipblasCdotc( handle, vector_size, x, incx, y, incy, result ) );
}
template <>
inline void hipblas_wrap::dot(
    size_t vector_size, const hipblasDoubleComplex *x, const hipblasDoubleComplex *y, hipblasDoubleComplex *result, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL( hipblasZdotc( handle, vector_size, x, incx, y, incy, result ) );
}
template <>
inline void hipblas_wrap::dot(
    size_t vector_size, const thrust::complex<float> *x, const thrust::complex<float> *y,
    thrust::complex<float> *result, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL(
        hipblasCdotc( handle, vector_size, (hipblasComplex *)x, incx, (hipblasComplex *)y, incy, (hipblasComplex *)result )
    );
}
template <>
inline void hipblas_wrap::dot(
    size_t vector_size, const thrust::complex<double> *x, const thrust::complex<double> *y,
    thrust::complex<double> *result, int incx, int incy
)
{
    HIPBLAS_SAFE_CALL( hipblasZdotc(
        handle, vector_size, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)y, incy, (hipblasDoubleComplex *)result
    ) );
}
//
template <>
inline void hipblas_wrap::asum( size_t vector_size, const float *x, float *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasSasum( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::asum( size_t vector_size, const double *x, double *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasDasum( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::asum(
    size_t vector_size, const hipblasComplex *x, typename hipblas_real_types::hipblas_real_type_hlp<hipblasComplex>::type *result,
    int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScasum( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::asum(
    size_t vector_size, const hipblasDoubleComplex *x,
    typename hipblas_real_types::hipblas_real_type_hlp<hipblasDoubleComplex>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasDzasum( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::asum(
    size_t vector_size, const thrust::complex<float> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<float>>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScasum( handle, vector_size, (hipblasComplex *)x, incx, result ) );
}
template <>
inline void hipblas_wrap::asum(
    size_t vector_size, const thrust::complex<double> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<double>>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasDzasum( handle, vector_size, (hipblasDoubleComplex *)x, incx, result ) );
}

//
template <>
inline void hipblas_wrap::norm2( size_t vector_size, const float *x, float *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasSnrm2( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::norm2( size_t vector_size, const double *x, double *result, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasDnrm2( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::norm2(
    size_t vector_size, const hipblasComplex *x, typename hipblas_real_types::hipblas_real_type_hlp<hipblasComplex>::type *result,
    int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScnrm2( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::norm2(
    size_t vector_size, const hipblasDoubleComplex *x,
    typename hipblas_real_types::hipblas_real_type_hlp<hipblasDoubleComplex>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasDznrm2( handle, vector_size, x, incx, result ) );
}
template <>
inline void hipblas_wrap::norm2(
    size_t vector_size, const thrust::complex<float> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<float>>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasScnrm2( handle, vector_size, (hipblasComplex *)x, incx, result ) );
}
template <>
inline void hipblas_wrap::norm2(
    size_t vector_size, const thrust::complex<double> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<double>>::type *result, int incx
)
{
    HIPBLAS_SAFE_CALL( hipblasDznrm2( handle, vector_size, (hipblasDoubleComplex *)x, incx, result ) );
}
//
template <>
inline void hipblas_wrap::scale( size_t vector_size, const float alpha, float *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasSscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const double alpha, double *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasDscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const hipblasComplex alpha, hipblasComplex *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasCscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const hipblasDoubleComplex alpha, hipblasDoubleComplex *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasZscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void
hipblas_wrap::scale( size_t vector_size, const thrust::complex<float> alpha, thrust::complex<float> *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasCscal( handle, vector_size, (hipblasComplex *)&alpha, (hipblasComplex *)x, incx ) );
}
template <>
inline void
hipblas_wrap::scale( size_t vector_size, const thrust::complex<double> alpha, thrust::complex<double> *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasZscal( handle, vector_size, (hipblasDoubleComplex *)&alpha, (hipblasDoubleComplex *)x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const float alpha, hipblasComplex *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasCsscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const double alpha, hipblasDoubleComplex *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasZdscal( handle, vector_size, &alpha, x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const float alpha, thrust::complex<float> *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasCsscal( handle, vector_size, &alpha, (hipblasComplex *)x, incx ) );
}
template <>
inline void hipblas_wrap::scale( size_t vector_size, const double alpha, thrust::complex<double> *x, int incx )
{
    HIPBLAS_SAFE_CALL( hipblasZdscal( handle, vector_size, &alpha, (hipblasDoubleComplex *)x, incx ) );
}
// aditional functions that are common
template <>
inline void hipblas_wrap::normalize( size_t vector_size, float *x, float *norm, int incx )
{
    norm2<float>( vector_size, (const float *)x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        float inorm = float( 1.0 ) / norm[0];
        scale<float>( vector_size, inorm, x, incx );
    }
}
template <>
inline void hipblas_wrap::normalize( size_t vector_size, double *x, double *norm, int incx )
{
    norm2<double>( vector_size, (const double *)x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        double inorm = float( 1.0 ) / norm[0];
        scale<double>( vector_size, inorm, x, incx );
    }
}
template <>
inline void hipblas_wrap::normalize(
    size_t vector_size, hipblasComplex *x, typename hipblas_real_types::hipblas_real_type_hlp<hipblasComplex>::type *norm, int incx
)
{
    norm2<hipblasComplex>( vector_size, (const hipblasComplex *)x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        float inorm = float( 1.0 ) / norm[0];
        scale<hipblasComplex>( vector_size, inorm, x, incx );
    }
}
template <>
inline void hipblas_wrap::normalize(
    size_t vector_size, hipblasDoubleComplex *x,
    typename hipblas_real_types::hipblas_real_type_hlp<hipblasDoubleComplex>::type *norm, int incx
)
{
    norm2<hipblasDoubleComplex>( vector_size, (const hipblasDoubleComplex *)x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        double inorm = double( 1.0 ) / norm[0];
        scale<hipblasDoubleComplex>( vector_size, inorm, x, incx );
    }
}
template <>
inline void hipblas_wrap::normalize(
    size_t vector_size, thrust::complex<float> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<float>>::type *norm, int incx
)
{
    norm2<thrust::complex<float>>( vector_size, x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        float inorm = float( 1.0 ) / norm[0];
        scale<thrust::complex<float>>( vector_size, inorm, x, incx );
    }
}
template <>
inline void hipblas_wrap::normalize(
    size_t vector_size, thrust::complex<double> *x,
    typename hipblas_real_types::hipblas_real_type_hlp<thrust::complex<double>>::type *norm, int incx
)
{
    norm2<thrust::complex<double>>( vector_size, x, norm, incx );
    if ( scalar_pointer_on_device )
    {
    }
    else
    {
        double inorm = double( 1.0 ) / norm[0];
        scale<thrust::complex<double>>( vector_size, inorm, x, incx );
    }
}


//level 2 BLAS specializations:

template <>
inline void hipblas_wrap::gemv(
    const char op, size_t RowA, const float *A, size_t ColA, size_t LDimA, float alpha, const float *x, float beta,
    float *y
)
{

    /*
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, 
                           hipblasOperation_t trans,
                           int m, 
                           int n,
                           const float *alpha,
                           const float *A, 
                           int lda,
                           const float *x, 
                           int incx,
                           const float *beta,
                           float *y, 
                           int incy)
*/
    HIPBLAS_SAFE_CALL(
        hipblasSgemv( handle, switch_operation_real( op ), RowA, ColA, &alpha, A, LDimA, x, 1, &beta, y, 1 )
    );
}

template <>
inline void hipblas_wrap::gemv(
    const char op, size_t RowA, const double *A, size_t ColA, size_t LDimA, double alpha, const double *x, double beta,
    double *y
)
{

    HIPBLAS_SAFE_CALL(
        hipblasDgemv( handle, switch_operation_real( op ), RowA, ColA, &alpha, A, LDimA, x, 1, &beta, y, 1 )
    );
}
template <>
inline void hipblas_wrap::gemv(
    const char op, size_t RowA, const thrust::complex<float> *A, size_t ColA, size_t LDimA,
    thrust::complex<float> alpha, const thrust::complex<float> *x, thrust::complex<float> beta,
    thrust::complex<float> *y
)
{

    HIPBLAS_SAFE_CALL( hipblasCgemv(
        handle, switch_operation_complex( op ), RowA, ColA, (const hipblasComplex *)&alpha, (const hipblasComplex *)A, LDimA,
        (const hipblasComplex *)x, 1, (const hipblasComplex *)&beta, (hipblasComplex *)y, 1
    ) );
}
template <>
inline void hipblas_wrap::gemv(
    const char op, size_t RowA, const thrust::complex<double> *A, size_t ColA, size_t LDimA,
    thrust::complex<double> alpha, const thrust::complex<double> *x, thrust::complex<double> beta,
    thrust::complex<double> *y
)
{

    HIPBLAS_SAFE_CALL( hipblasZgemv(
        handle, switch_operation_complex( op ), RowA, ColA, (const hipblasDoubleComplex *)&alpha, (const hipblasDoubleComplex *)A,
        LDimA, (const hipblasDoubleComplex *)x, 1, (const hipblasDoubleComplex *)&beta, (hipblasDoubleComplex *)y, 1
    ) );
}

//level 3 BLAS specializations:
template <>
inline void hipblas_wrap::gemm(
    char opA, char opB, size_t RowA, size_t ColBC, size_t ColARowB, float alpha, const float *A, size_t LDimA,
    const float *B, size_t LDimB, float beta, float *C, size_t LDimC
)
{
    /*
 C = α op ( A ) op ( B ) + β C 

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,
                           hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)

    m - number of rows of matrix op(A) and C.

    n - number of columns of matrix op(B) and C.
    
    k - number of columns of op(A) and rows of op(B). 

    lda - leading dimension of two-dimensional array used to store the matrix A. 

    ldb - leading dimension of two-dimensional array used to store matrix B. 

    ldc - leading dimension of a two-dimensional array used to store the matrix C. 

*/

    HIPBLAS_SAFE_CALL( hipblasSgemm(
        handle, switch_operation_real( opA ), switch_operation_real( opB ), RowA, ColBC, ColARowB, &alpha, A, LDimA, B,
        LDimB, &beta, C, LDimC
    ) );
}

template <>
inline void hipblas_wrap::gemm(
    char opA, char opB, size_t RowA, size_t ColBC, size_t ColARowB, double alpha, const double *A, size_t LDimA,
    const double *B, size_t LDimB, double beta, double *C, size_t LDimC
)
{
    HIPBLAS_SAFE_CALL( hipblasDgemm(
        handle, switch_operation_real( opA ), switch_operation_real( opB ), RowA, ColBC, ColARowB, &alpha, A, LDimA, B,
        LDimB, &beta, C, LDimC
    ) );
}
template <>
inline void hipblas_wrap::gemm(
    char opA, char opB, size_t RowA, size_t ColBC, size_t ColARowB, thrust::complex<float> alpha,
    const thrust::complex<float> *A, size_t LDimA, const thrust::complex<float> *B, size_t LDimB,
    thrust::complex<float> beta, thrust::complex<float> *C, size_t LDimC
)
{
    HIPBLAS_SAFE_CALL( hipblasCgemm(
        handle, switch_operation_complex( opA ), switch_operation_complex( opB ), RowA, ColBC, ColARowB,
        (const hipblasComplex *)&alpha, (const hipblasComplex *)A, LDimA, (const hipblasComplex *)B, LDimB, (const hipblasComplex *)&beta,
        (hipblasComplex *)C, LDimC
    ) );
}
template <>
inline void hipblas_wrap::gemm(
    char opA, char opB, size_t RowA, size_t ColBC, size_t ColARowB, thrust::complex<double> alpha,
    const thrust::complex<double> *A, size_t LDimA, const thrust::complex<double> *B, size_t LDimB,
    thrust::complex<double> beta, thrust::complex<double> *C, size_t LDimC
)
{
    HIPBLAS_SAFE_CALL( hipblasZgemm(
        handle, switch_operation_complex( opA ), switch_operation_complex( opB ), RowA, ColBC, ColARowB,
        (const hipblasDoubleComplex *)&alpha, (const hipblasDoubleComplex *)A, LDimA, (const hipblasDoubleComplex *)B, LDimB,
        (const hipblasDoubleComplex *)&beta, (hipblasDoubleComplex *)C, LDimC
    ) );
}


template <>
inline void hipblas_wrap::trsm(
    const char sideA, const char isA_UorL, const char opA, bool unit_diag_A, size_t RowBColA, size_t ColsB,
    const double alpha, const double *A, size_t LDimA, double *B, size_t LDimB
)
{

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    if ( ( sideA == 'r' ) || ( sideA == 'R' ) )
    {
        side = HIPBLAS_SIDE_RIGHT;
    }
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    if ( ( isA_UorL == 'u' ) || ( isA_UorL == 'U' ) )
    {
        uplo = HIPBLAS_FILL_MODE_UPPER;
    }
    hipblasDiagType_t diag = HIPBLAS_DIAG_NON_UNIT;
    if ( unit_diag_A )
    {
        diag = HIPBLAS_DIAG_UNIT;
    }

    HIPBLAS_SAFE_CALL( hipblasDtrsm(
        handle, side, uplo, switch_operation_real( opA ), diag, int( RowBColA ), int( ColsB ), &alpha, A, int( LDimA ),
        B, int( LDimB )
    )

    );
}
template <>
inline void hipblas_wrap::trsm(
    const char sideA, const char isA_UorL, const char opA, bool unit_diag_A, size_t RowBColA, size_t ColsB,
    const float alpha, const float *A, size_t LDimA, float *B, size_t LDimB
)
{

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    if ( ( sideA == 'r' ) || ( sideA == 'R' ) )
    {
        side = HIPBLAS_SIDE_RIGHT;
    }
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    if ( ( isA_UorL == 'u' ) || ( isA_UorL == 'U' ) )
    {
        uplo = HIPBLAS_FILL_MODE_UPPER;
    }
    hipblasDiagType_t diag = HIPBLAS_DIAG_NON_UNIT;
    if ( unit_diag_A )
    {
        diag = HIPBLAS_DIAG_UNIT;
    }

    HIPBLAS_SAFE_CALL( hipblasStrsm(
        handle, side, uplo, switch_operation_real( opA ), diag, int( RowBColA ), int( ColsB ), &alpha, A, int( LDimA ),
        B, int( LDimB )
    )

    );
}
template <>
inline void hipblas_wrap::trsm(
    const char sideA, const char isA_UorL, const char opA, bool unit_diag_A, size_t RowBColA, size_t ColsB,
    const thrust::complex<float> alpha, const thrust::complex<float> *A, size_t LDimA, thrust::complex<float> *B,
    size_t LDimB
)
{

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    if ( ( sideA == 'r' ) || ( sideA == 'R' ) )
    {
        side = HIPBLAS_SIDE_RIGHT;
    }
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    if ( ( isA_UorL == 'u' ) || ( isA_UorL == 'U' ) )
    {
        uplo = HIPBLAS_FILL_MODE_UPPER;
    }
    hipblasDiagType_t diag = HIPBLAS_DIAG_NON_UNIT;
    if ( unit_diag_A )
    {
        diag = HIPBLAS_DIAG_UNIT;
    }

    HIPBLAS_SAFE_CALL( hipblasCtrsm(
        handle, side, uplo, switch_operation_real( opA ), diag, int( RowBColA ), int( ColsB ),
        (const hipblasComplex *)&alpha, (const hipblasComplex *)A, int( LDimA ), (hipblasComplex *)B, int( LDimB )
    )

    );
}
template <>
inline void hipblas_wrap::trsm(
    const char sideA, const char isA_UorL, const char opA, bool unit_diag_A, size_t RowBColA, size_t ColsB,
    const thrust::complex<double> alpha, const thrust::complex<double> *A, size_t LDimA, thrust::complex<double> *B,
    size_t LDimB
)
{

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    if ( ( sideA == 'r' ) || ( sideA == 'R' ) )
    {
        side = HIPBLAS_SIDE_RIGHT;
    }
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    if ( ( isA_UorL == 'u' ) || ( isA_UorL == 'U' ) )
    {
        uplo = HIPBLAS_FILL_MODE_UPPER;
    }
    hipblasDiagType_t diag = HIPBLAS_DIAG_NON_UNIT;
    if ( unit_diag_A )
    {
        diag = HIPBLAS_DIAG_UNIT;
    }

    HIPBLAS_SAFE_CALL( hipblasZtrsm(
        handle, side, uplo, switch_operation_real( opA ), diag, int( RowBColA ), int( ColsB ),
        (const hipblasDoubleComplex *)&alpha, (const hipblasDoubleComplex *)A, int( LDimA ), (hipblasDoubleComplex *)B, int( LDimB )
    )

    );
}


template <>
inline void hipblas_wrap::geam(
    const char opA, size_t RowAC, size_t ColBC, const float alpha, const float *A, size_t LDimA, const float beta,
    float *B, size_t LDimB, float *C, size_t LDimC
)
{

    HIPBLAS_SAFE_CALL( hipblasSgeam(
        handle, switch_operation_real( opA ), HIPBLAS_OP_N, int( RowAC ), int( ColBC ), (const float *)&alpha,
        (const float *)A, int( LDimA ), (const float *)&beta, (const float *)B, int( LDimB ), C, int( LDimC )
    ) );
}

template <>
inline void hipblas_wrap::geam(
    const char opA, size_t RowAC, size_t ColBC, const double alpha, const double *A, size_t LDimA, const double beta,
    double *B, size_t LDimB, double *C, size_t LDimC
)
{

    HIPBLAS_SAFE_CALL( hipblasDgeam(
        handle, switch_operation_real( opA ), HIPBLAS_OP_N, int( RowAC ), int( ColBC ), (const double *)&alpha,
        (const double *)A, int( LDimA ), (const double *)&beta, (const double *)B, int( LDimB ), C, int( LDimC )
    ) );
}

template <>
inline void hipblas_wrap::geam(
    const char opA, size_t RowAC, size_t ColBC, const thrust::complex<float> alpha, const thrust::complex<float> *A,
    size_t LDimA, const thrust::complex<float> beta, thrust::complex<float> *B, size_t LDimB, thrust::complex<float> *C,
    size_t LDimC
)
{

    HIPBLAS_SAFE_CALL( hipblasCgeam(
        handle, switch_operation_real( opA ), HIPBLAS_OP_N, int( RowAC ), int( ColBC ), (const hipblasComplex *)&alpha,
        (const hipblasComplex *)A, int( LDimA ), (const hipblasComplex *)&beta, (const hipblasComplex *)B, int( LDimB ),
        (hipblasComplex *)C, int( LDimC )
    ) );
}

template <>
inline void hipblas_wrap::geam(
    const char opA, size_t RowAC, size_t ColBC, const thrust::complex<double> alpha, const thrust::complex<double> *A,
    size_t LDimA, const thrust::complex<double> beta, thrust::complex<double> *B, size_t LDimB,
    thrust::complex<double> *C, size_t LDimC
)
{

    HIPBLAS_SAFE_CALL( hipblasZgeam(
        handle, switch_operation_real( opA ), HIPBLAS_OP_N, int( RowAC ), int( ColBC ), (const hipblasDoubleComplex *)&alpha,
        (const hipblasDoubleComplex *)A, int( LDimA ), (const hipblasDoubleComplex *)&beta, (const hipblasDoubleComplex *)B,
        int( LDimB ), (hipblasDoubleComplex *)C, int( LDimC )
    ) );
}

} // namespace scfd


#endif