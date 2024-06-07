#ifndef __SCFD_MEMORY_THRUST_PTR_UNIFIED_H__
#define __SCFD_MEMORY_THRUST_PTR_UNIFIED_H__

#if CUDART_VERSION >= 11040
#include <thrust/universal_ptr.h>
#else
#include <thrust/device_ptr.h>
#endif
#include "thrust_ptr.h"
#include "unified.h"

namespace scfd
{
namespace memory
{

#if CUDART_VERSION >= 11040
template<class T>
struct thrust_ptr<scfd::memory::unified,T>
{
    using type = thrust::universal_ptr<T>;
    static type cast(T *p)
    {
        return ::thrust::universal_ptr<T>(p);
    }
};
#else
template<class T>
struct thrust_ptr<scfd::memory::unified,T>
{
    using type = thrust::device_ptr<T>;
    static type cast(T *p)
    {
        return ::thrust::device_ptr<T>(p);
    }
};
#endif

} /// namespace memory
} /// namespace scfd

#endif
