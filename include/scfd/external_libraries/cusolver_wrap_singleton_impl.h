#ifndef __SCFD_CUSOLVER_WRAP_SINGLETON_IMPL_H__
#define __SCFD_CUSOLVER_WRAP_SINGLETON_IMPL_H__

#include "cusolver_wrap.h"

namespace scfd
{

template<>
cusolver_wrap *manual_init_singleton<cusolver_wrap>::inst_ = nullptr;

}

#endif
