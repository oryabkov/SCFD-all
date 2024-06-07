#ifndef __SCFD_MEMORY_THRUST_PTR_CUDA_H__
#define __SCFD_MEMORY_THRUST_PTR_CUDA_H__

#include <thrust/device_ptr.h>
#include "thrust_ptr.h"
#include "cuda.h"

namespace scfd
{
namespace memory
{

template<class T>
struct thrust_ptr<cuda_device,T>
{
    using type = thrust::device_ptr<T>;
    static type cast(T *p)
    {
        return ::thrust::device_pointer_cast(p);
    }
};

template<class T>
struct thrust_ptr<cuda_host,T>
{
    using type = T*;
    static type cast(T *p)
    {
        return p;
    }
};

} /// namespace memory
} /// namespace scfd

#endif
