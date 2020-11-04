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

#ifndef _SCFD_UTILS_TIMER_EVENT_H__
#define _SCFD_UTILS_TIMER_EVENT_H__

namespace scfd
{
namespace utils
{

/// Noncopyable and Nonmoveable
struct timer_event
{
    timer_event() = default;
    /// NOTE as I understand it also forbids move semantics as well
    timer_event(const timer_event &e) = delete;
    timer_event &operator=(const timer_event &e) = delete;

    virtual ~timer_event() = default;

    virtual void    record() = 0;
    virtual double  elapsed_time(const timer_event &e0)const = 0;
};

}

}

#endif
