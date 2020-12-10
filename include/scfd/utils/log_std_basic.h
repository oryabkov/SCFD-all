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

#ifndef __SCFD_UTILS_LOG_STD_BASIC_H__
#define __SCFD_UTILS_LOG_STD_BASIC_H__

/// For SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE description see log_cformatted.h

#include <string>
#include <exception>
#include <stdexcept>
#include <iostream>
#include "log_msg_type.h"
#if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
#include <mutex>
#include <atomic>
#endif

namespace scfd
{
namespace utils
{

class log_std_basic
{
public:
    using log_msg_type = utils::log_msg_type;

public:
    log_std_basic() : log_lev(1) {}

    void msg(const std::string &s, log_msg_type mt = log_msg_type::INFO, int _log_lev = 1)
    {
        if ((mt != log_msg_type::ERROR)&&(_log_lev > log_lev)) return;

        #if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
        std::lock_guard<std::mutex> locker(mtx_);
        #endif

        //TODO
        if ((mt == log_msg_type::INFO)||(mt == log_msg_type::INFO_ALL))
            std::cout << "INFO:    " << s << std::endl;
        else if (mt == log_msg_type::WARNING)
            std::cout << "WARNING: " << s << std::endl;
        else if (mt == log_msg_type::ERROR)
            std::cout << "ERROR:   " << s << std::endl;
        else if (mt == log_msg_type::DEBUG)
            std::cout << "DEBUG:   " << s << std::endl;
        else
            throw std::logic_error("log_std_basic::log: wrong t_msg_type argument");
    }
    void set_verbosity(int _log_lev = 1) { log_lev = _log_lev; }

private:
    /// NOTE we use atomic for verbosity to: 
    /// 1. garantee thread safety
    /// 2. don't use mtx_ for log_lev protection (to not slower output more)
    #if SCFD_UTILS_LOG_GARANTEE_THREAD_SAFE==1
    std::atomic<int>    log_lev;
    std::mutex          mtx_;
    #else
    int                 log_lev;
    #endif
};

}

}

#endif
