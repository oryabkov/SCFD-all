#ifndef __SCFD_MEMORY_THRUST_PTR_HOST_H__
#define __SCFD_MEMORY_THRUST_PTR_HOST_H__

#include "thrust_ptr.h"
#include "host.h"

namespace scfd
{
namespace memory
{

template<class T>
struct thrust_ptr<host,T>
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
