#ifndef __ARRAY_THRUST_POINTER_CAST_H__
#define __ARRAY_THRUST_POINTER_CAST_H__

/// TODO make separate file for cuda specialization
#include <scfd/memory/host.h>
#include <scfd/memory/cuda.h>
#if defined(STOKES_PORUS_3D_PLATFORM_CUDA_UNIFIED)
#include <scfd/memory/unified.h>
#endif
#include <scfd/memory/thrust_ptr.h>

namespace detail {

template<class Array>
class array_thrust_ptr
{
    using value_t = typename Array::value_type;
    using memory_t = typename Array::memory_type;
    using memory_thrust_ptr_t = scfd::memory::thrust_ptr<memory_t,value_t>;
public:
    using type = typename memory_thrust_ptr_t::type;
    static type cast(value_t *p)
    {
        return memory_thrust_ptr_t::cast(p);
    }
};

template<class Array>
typename array_thrust_ptr<Array>::type
array_thrust_begin(const Array array)
{
    return array_thrust_ptr<Array>::cast(array.raw_ptr());
}

template<class Array>
typename array_thrust_ptr<Array>::type
array_thrust_end(const Array array)
{
    return array_thrust_ptr<Array>::cast(array.raw_ptr()) + array.total_size();
}

} /// namespace detail

#endif

