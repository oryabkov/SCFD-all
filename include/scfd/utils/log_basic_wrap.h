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

#ifndef __SCFD_UTILS_LOG_BASIC_WRAP_H__
#define __SCFD_UTILS_LOG_BASIC_WRAP_H__

#include <string>

/**
* Class needed, if we want to wrap someone's other LogBasic in our log_cformatted<>
* and need to expose ability to pass their own LogBasic instance through set_log-like 
* method or constructor.
*/

namespace scfd
{
namespace utils
{

/// LogBasic concept see in LOG_CONCEPTS.txt
template<class LogBasic>
class log_basic_wrap
{
public:
    using log_msg_type = typename LogBasic::log_msg_type;
    using log_basic_type = LogBasic;

public:
    log_basic_wrap(LogBasic *log_basic = NULL) : log_basic_(log_basic)
    {
    }

    void set_log_basic(LogBasic *log_basic)
    {
        log_basic_ = log_basic;
    }
    LogBasic *get_log_basic()const 
    {
        return log_basic_;
    }

    void msg(const std::string &s, log_msg_type mt = log_msg_type::INFO, int log_lev = 1)
    {
        if (log_basic_ != NULL) log_basic_->msg(s, mt, log_lev);
    }
    void set_verbosity(int log_lev = 1) 
    { 
        if (log_basic_ != NULL) log_basic_->set_verbosity(log_lev);
    }

private:
    LogBasic    *log_basic_;
};

}

}

#endif
