// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_LOGGED_OBJ_BASE_H__
#define __SCFD_UTILS_LOGGED_OBJ_BASE_H__

#include <cstdarg>
#include <string>

/// logged_obj_base itself supposed to satisfy Log (at least LogCFormatted) concept
// TODO add neccessary methods (msg, v_*-output methods)

namespace scfd
{
namespace utils
{

template<class Log>
class logged_obj_base
{
protected:
    Log             *log_;
    int             obj_log_lev_;
    std::string     log_msg_prefix_;
public:
    logged_obj_base(Log *log__ = NULL, int obj_log_lev__ = 0, const std::string &log_msg_prefix__ = "") : 
        log_(log__),obj_log_lev_(obj_log_lev__),log_msg_prefix_(log_msg_prefix__) {}

    void                set_log(Log *log__) { log_ = log__; }
    Log                 *get_log()const { return log_; }
    void                set_obj_log_lev(int obj_log_lev__) { obj_log_lev_ = obj_log_lev__; }
    int                 get_obj_log_lev()const { return obj_log_lev_; }
    void                set_log_msg_prefix(const std::string &log_msg_prefix__) 
    { 
        log_msg_prefix_ = log_msg_prefix__; 
    }
    const std::string   &get_log_msg_prefix()const { return log_msg_prefix_; }

    void info(int log_lev_, const std::string &s)const
    {
        if (log_ != NULL) log_->info(obj_log_lev_ + log_lev_, log_msg_prefix_ + s);
    }
    void info_all(int log_lev_, const std::string &s)const
    {
        if (log_ != NULL) log_->info_all(obj_log_lev_ + log_lev_, log_msg_prefix_ + s);
    }
    void warning(int log_lev_, const std::string &s)const
    {
        if (log_ != NULL) log_->warning(obj_log_lev_ + log_lev_, log_msg_prefix_ + s);
    }
    void error(int log_lev_, const std::string &s)const
    {
        if (log_ != NULL) log_->error(obj_log_lev_ + log_lev_, log_msg_prefix_ + s);
    }
    void debug(int log_lev_, const std::string &s)const
    {
        if (log_ != NULL) log_->debug(obj_log_lev_ + log_lev_, log_msg_prefix_ + s);
    }
    void info(const std::string &s)const
    {
        info(1, s);
    }
    void info_all(const std::string &s)const
    {
        info_all(1, s);
    }
    void warning(const std::string &s)const
    {
        warning(1, s);
    }
    void error(const std::string &s)const
    {
        error(1, s);
    }
    void debug(const std::string &s)const
    {
        debug(1, s);
    }

    #define LOGGED_OBJ_BASE__FORMATTED_OUT__(METHOD_NAME, LOG_LEV)              \
        if (log_ == NULL) return;                                               \
        va_list arguments;                                                      \
        va_start ( arguments, s );                                              \
        log_->METHOD_NAME(LOG_LEV, log_msg_prefix_ + s, arguments);             \
        va_end ( arguments );    
    void info_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_f, obj_log_lev_ + log_lev_)
    }
    void info_all_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_all_f, obj_log_lev_ + log_lev_)
    }
    void warning_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_warning_f, obj_log_lev_ + log_lev_)
    }
    void error_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_error_f, obj_log_lev_ + log_lev_)
    }
    void debug_f(int log_lev_, const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_debug_f, obj_log_lev_ + log_lev_)
    }
    void info_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_f, obj_log_lev_ + 1)
    }
    void info_all_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_info_all_f, obj_log_lev_ + 1)
    }
    void warning_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_warning_f, obj_log_lev_ + 1)
    }
    void error_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_error_f, obj_log_lev_ + 1)
    }
    void debug_f(const std::string &s, ...)const
    {
        LOGGED_OBJ_BASE__FORMATTED_OUT__(v_debug_f, obj_log_lev_ + 1)
    }
    #undef LOGGED_OBJ_BASE__FORMATTED_OUT__
};

}

}

#endif 