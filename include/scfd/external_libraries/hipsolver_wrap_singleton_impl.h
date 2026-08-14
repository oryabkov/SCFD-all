#ifndef __SCFD_HIPSOLVER_WRAP_SINGLETON_IMPL_H__
#define __SCFD_HIPSOLVER_WRAP_SINGLETON_IMPL_H__

#include "hipsolver_wrap.h"

namespace scfd
{

template <>
inline hipsolver_wrap *utils::manual_init_singleton<hipsolver_wrap>::inst_ = nullptr;

}

#endif
