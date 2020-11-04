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

#ifndef __SCFD_UTILS_LOG_BASIC_CFORMATTED_WRAP_H__
#define __SCFD_UTILS_LOG_BASIC_CFORMATTED_WRAP_H__

#include "log_basic_wrap.h"
#include "log_cformatted.h"

/**
* TODO
*/

namespace scfd
{
namespace utils
{

/// LogBasic concept see in LOG_CONCEPTS.txt
template<class LogBasic>
class log_basic_cformatted_wrap : public log_cformatted<log_basic_wrap<LogBasic>>
{
public:
    log_basic_cformatted_wrap(LogBasic *log_basic = NULL) : log_cformatted<log_basic_wrap<LogBasic>>(log_basic)
    {
    }
};

}

}

#endif
