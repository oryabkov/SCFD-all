// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_UTILS_LOG_CFORMATTED_BASE_H__
#define __SCFD_UTILS_LOG_CFORMATTED_BASE_H__

#include <cstdarg>
#include <cstdio>

/**
* SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE macro set manages
* whether mutexes are used in log* classes to garantee
* thread safety.
* SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1 means yes,
* SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==0 means no
*/ 

#if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
#include <mutex>
#endif

namespace scfd
{
namespace utils
{

/**
* LogBasic concept see in LOG_CONCEPTS.txt
*/ 

template<class LogBasic, std::size_t BufSize = 512>
class log_cformatted : public LogBasic
{
public:
    using LogBasic::LogBasic;
    //using typename LogBasic::log_msg_type;
    /// NOTE for some reason above using version didnot work with nvcc (8.0) however worked with g++
    using log_msg_type = typename LogBasic::log_msg_type;
    using LogBasic::msg;
    using LogBasic::set_verbosity;

public:
    void info(const std::string &s)
    {
        msg(s, log_msg_type::INFO, 1);
    }
    void info_all(const std::string &s)
    {
        msg(s, log_msg_type::INFO_ALL, 1);
    }
    void warning(const std::string &s)
    {
        msg(s, log_msg_type::WARNING, 1);
    }
    void error(const std::string &s)
    {
        msg(s, log_msg_type::ERROR, 1);
    }
    void debug(const std::string &s)
    {
        msg(s, log_msg_type::DEBUG, 1);
    }
    void info(int _log_lev, const std::string &s)
    {
        msg(s, log_msg_type::INFO, _log_lev);
    }
    void info_all(int _log_lev, const std::string &s)
    {
        msg(s, log_msg_type::INFO_ALL, _log_lev);
    }
    void warning(int _log_lev, const std::string &s)
    {
        msg(s, log_msg_type::WARNING, _log_lev);
    }
    void error(int _log_lev, const std::string &s)
    {
        msg(s, log_msg_type::ERROR, _log_lev);
    }
    void debug(int _log_lev, const std::string &s)
    {
        msg(s, log_msg_type::DEBUG, _log_lev);
    }
    
    #if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
    #define SCFD_UTILS_LOG__FORMATTED_LOCK std::lock_guard<std::mutex> locker(mtx_);
    #else
    #define SCFD_UTILS_LOG__FORMATTED_LOCK
    #endif

    #define LOG__FORMATTED_OUT_V__(METHOD_NAME,LOG_LEV)   \
        SCFD_UTILS_LOG__FORMATTED_LOCK                    \
        vsnprintf(buf, BufSize, s.c_str(), arguments);    \
        METHOD_NAME(LOG_LEV, std::string(buf));
    void v_info_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info, _log_lev)
    }
    void v_info_all_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info_all, _log_lev)
    }
    void v_warning_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(warning, _log_lev)
    }
    void v_error_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(error, _log_lev)
    }
    void v_debug_f(int _log_lev, const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(debug, _log_lev)
    }
    void v_info_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info, 1)
    }
    void v_info_all_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(info_all, 1)
    }
    void v_warning_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(warning, 1)
    }
    void v_error_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(error, 1)
    }
    void v_debug_f(const std::string &s, va_list arguments)
    {
        LOG__FORMATTED_OUT_V__(debug, 1)
    }
    #undef LOG__FORMATTED_OUT_V__ 


    #define LOG__FORMATTED_OUT__(METHOD_NAME,LOG_LEV)   \
        SCFD_UTILS_LOG__FORMATTED_LOCK                  \
        va_list arguments;                              \
        va_start ( arguments, s );                      \
        vsnprintf(buf, BufSize, s.c_str(), arguments);  \
        METHOD_NAME(LOG_LEV, std::string(buf));         \
        va_end ( arguments );   
    void info_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info, _log_lev)
    }
    void info_all_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info_all, _log_lev)
    }
    void warning_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(warning, _log_lev)
    }
    void error_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(error, _log_lev)
    }
    void debug_f(int _log_lev, const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(debug, _log_lev)
    }
    void info_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info, 1)
    }
    void info_all_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(info_all, 1)
    }
    void warning_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(warning, 1)
    }
    void error_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(error, 1)
    }
    void debug_f(const std::string &s, ...)
    {
        LOG__FORMATTED_OUT__(debug, 1)
    }
    #undef LOG__FORMATTED_OUT__

private:
    char        buf[BufSize];
    #if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
    std::mutex  mtx_;
    #endif


};

}
}

#endif
