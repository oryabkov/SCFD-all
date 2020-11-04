// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_CONSTANT_DATA_H__
#define __SCFD_UTILS_CONSTANT_DATA_H__

//TODO hmm it's hack to overcome problem with my nvcc toolkit 3.2 where __CUDACC__ macro is defined but not __NVCC__
//need to be deleted (figure out from which version there is __NVCC__ macro)
//another reason to think about manual customization
#ifdef __CUDACC__
#ifndef __NVCC__
#define __NVCC__
#endif
#endif

#include <cstdlib>   
#include <cstring> 
#ifdef __NVCC__
#include "cuda_safe_call.h"
#endif

//NOTE like simple CUDA __constant__ variables this 'constant buffer' shares the same visbility principle:
//it's only visible inside current compiling module

//ISSUE didnot figure out anything clever then just make two copies in defines - one for cuda case and one for pure c++
//ISSUE like in tensor_fields it whould be better to create special macro to manage cuda/noncuda behaviour then 
//simply looking for __NVCC__ define


template<typename T> struct utils_argument_type;
template<typename T, typename U> struct utils_argument_type<T(U)> { typedef U type; };

#define __CONSTANT_BUFFER__CTASTR2(pre,post) pre ## post
#define __CONSTANT_BUFFER__CTASTR(pre,post) __CONSTANT_BUFFER__CTASTR2(pre,post)


#ifdef __NVCC__

#ifdef __CUDA_ARCH__
#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                                                        \
    __device__ __host__ utils_argument_type<void(buf_type)>::type    &buf_name()                                      \
    {                                                                                                                 \
        return  reinterpret_cast<utils_argument_type<void(buf_type)>::type&>(__CONSTANT_BUFFER__CTASTR(__,buf_name)); \
    }
#else
#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                                                           \
    __device__ __host__ static utils_argument_type<void(buf_type)>::type    &buf_name()                                  \
    {                                                                                                                    \
        return  reinterpret_cast<utils_argument_type<void(buf_type)>::type&>(__CONSTANT_BUFFER__CTASTR(__h_,buf_name));  \
    }
#endif

#define DEFINE_CONSTANT_BUFFER(buf_type, buf_name)                                                  \
    __constant__ struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                                 \
    {                                                                                               \
        long long buf[sizeof(utils_argument_type<void(buf_type)>::type)/sizeof(long long)+1];       \
    } __CONSTANT_BUFFER__CTASTR(__,buf_name);                                                       \
    __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);    \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)

#define DECLARE_CONSTANT_BUFFER(buf_type, buf_name)                                                 \
    extern __constant__ struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                          \
    {                                                                                               \
        long long buf[sizeof(utils_argument_type<void(buf_type)>::type)/sizeof(long long)+1];       \
    } __CONSTANT_BUFFER__CTASTR(__,buf_name);                                                       \
    extern __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);    \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)

#define COPY_TO_CONSTANT_BUFFER(buf_name, data)                                                                                         \
    do {                                                                                                                                \
        __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)     _data;                                                                          \
        memcpy( &_data, &data, sizeof(data) );                                                                                          \
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(__CONSTANT_BUFFER__CTASTR(__,buf_name), &_data, sizeof(_data), 0, cudaMemcpyHostToDevice) ); \
        memcpy( &(__CONSTANT_BUFFER__CTASTR(__h_,buf_name)), &data, sizeof(data) );                                                     \
    } while (0)

#define COPY_ARRAY_TO_CONSTANT_BUFFER(buf_name, data_ptr, data_sz)                                                                      \
    do {                                                                                                                                \
        __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)     _data;                                                                          \
        memcpy( &_data, data_ptr, data_sz );                                                                                          \
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(__CONSTANT_BUFFER__CTASTR(__,buf_name), &_data, sizeof(_data), 0, cudaMemcpyHostToDevice) ); \
        memcpy( &(__CONSTANT_BUFFER__CTASTR(__h_,buf_name)), data_ptr, data_sz );                                                     \
    } while (0)

#else

#define DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)                                                           \
    static utils_argument_type<void(buf_type)>::type    &buf_name()                                                      \
    {                                                                                                                    \
        return  reinterpret_cast<utils_argument_type<void(buf_type)>::type&>(__CONSTANT_BUFFER__CTASTR(__h_,buf_name));  \
    }

#define DEFINE_CONSTANT_BUFFER(buf_type, buf_name)                                                  \
    struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                                              \
    {                                                                                               \
        long long buf[sizeof(utils_argument_type<void(buf_type)>::type)/sizeof(long long)+1];       \
    };                                                                                              \
    __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);           \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)

#define DECLARE_CONSTANT_BUFFER(buf_type, buf_name)                                                  \
    struct __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name)                                              \
    {                                                                                               \
        long long buf[sizeof(utils_argument_type<void(buf_type)>::type)/sizeof(long long)+1];       \
    };                                                                                              \
    extern __CONSTANT_BUFFER__CTASTR(__t_buf,buf_name) __CONSTANT_BUFFER__CTASTR(__h_,buf_name);    \
    DEFINE_CONSTANT_BUFFER_ACCESS_FUNC(buf_type, buf_name)


#define COPY_TO_CONSTANT_BUFFER(buf_name, data)                                                     \
    do {                                                                                            \
        memcpy( &(__CONSTANT_BUFFER__CTASTR(__h_,buf_name)), &data, sizeof(data) );                 \
    } while (0)

#endif

#endif
