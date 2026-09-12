#ifndef __SCFD_UTILS_PROFILING_H__
#define __SCFD_UTILS_PROFILING_H__

#include <scfd/utils/profiler.h>

#define SCFD_GLOBAL_PROFILING(timer_event_type) \
    using current_prof = scfd::utils::profiler<timer_event_type>; \
    namespace scfd { namespace utils { \
    template<> \
    inline current_prof *manual_init_singleton<current_prof>::inst_ = nullptr; \
    } }

#define SCFD_CONCAT_(a,b) a##b
#define SCFD_CONCAT(a,b) SCFD_CONCAT_(a,b)

#ifdef SCFD_ENABLE_PROFILING

#define SCFD_PROFILING_TIC(name) \
    current_prof::inst().tic(name);
#define SCFD_PROFILING_TOC(name) \
    current_prof::inst().toc(name);
#define SCFD_PROFILING_SCOPED_TIC(name) \
    auto SCFD_CONCAT(scfd_scoped_ticker_,__LINE__) = current_prof::inst().scoped_tic(name);
#define SCFD_PROFILING_SCOPED_TIC_PRINT(name, log) \
    auto SCFD_CONCAT(scfd_scoped_ticker_print_,__LINE__) = current_prof::inst().scoped_tic_print(name, log);
#else
#define SCFD_PROFILING_TIC(name)
#define SCFD_PROFILING_TOC(name)
#define SCFD_PROFILING_SCOPED_TIC(name)
#define SCFD_PROFILING_SCOPED_TIC_PRINT(name, log)
#endif

#endif
