#ifndef __SCFD_MEMORY_THRUST_PTR_H__
#define __SCFD_MEMORY_THRUST_PTR_H__

namespace scfd
{
namespace memory
{

template<class Memory, class T>
struct thrust_ptr
{
};

template<class Memory, class T>
typename thrust_ptr<Memory,T>::type thrust_ptr_cast(T *p)
{
    return thrust_ptr<Memory,T>::cast(p);
}

} /// namespace memory
} /// namespace scfd

#endif
