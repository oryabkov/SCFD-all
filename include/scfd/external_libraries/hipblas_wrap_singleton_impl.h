#ifndef __SCFD_HIPBLAS_WRAP_SINGLETON_IMPL_H__
#define __SCFD_HIPBLAS_WRAP_SINGLETON_IMPL_H__

#include "hipblas_wrap.h"

namespace scfd
{

template <>
inline hipblas_wrap *utils::manual_init_singleton<hipblas_wrap>::inst_ = nullptr;

}

#endif
