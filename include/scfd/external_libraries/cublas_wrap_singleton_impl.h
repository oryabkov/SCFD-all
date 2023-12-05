#ifndef __SCFD_CUBLAS_WRAP_SINGLETON_IMPL_H__
#define __SCFD_CUBLAS_WRAP_SINGLETON_IMPL_H__

#include "cublas_wrap.h"

namespace scfd
{

template<>
cublas_wrap *utils::manual_init_singleton<cublas_wrap>::inst_ = nullptr;

}

#endif
