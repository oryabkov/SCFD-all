#ifndef __SCFD_UTILS_PROFILING_H__
#define __SCFD_UTILS_PROFILING_H__

#include <scfd/utils/profiler.h>

namespace scfd { namespace utils {
template<>
inline current_prof *manual_init_singleton<current_prof>::inst_ = nullptr;
} }

#ifdef SCFD_ENABLE_PROFILING

#define SCFD_CONCAT_(a,b) a##b
#define SCFD_CONCAT(a,b) SCFD_CONCAT_(a,b)

#define SCFD_PLATFORM_TIC(name) \
    current_prof::inst().tic(name);
#define SCFD_PLATFORM_TOC(name) \
    current_prof::inst().toc(name);
#define SCFD_PLATFORM_SCOPED_TIC(name) \
    auto SCFD_CONCAT(scfd_scoped_ticker_,__LINE__) = current_prof::inst().scoped_tic(name);
#define SCFD_PLATFORM_SCOPED_TIC_PRINT(name, log) \
    auto SCFD_CONCAT(scfd_scoped_ticker_print_,__LINE__) = current_prof::inst().scoped_tic_print(name, log);
#else
#define SCFD_PLATFORM_TIC(name)
#define SCFD_PLATFORM_TOC(name)
#define SCFD_PLATFORM_SCOPED_TIC(name)
#define SCFD_PLATFORM_SCOPED_TIC_PRINT(name, log)
#endif

#endif
