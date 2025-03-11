// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef _SCFD_UTILS_HIP_TIMER_EVENT_H__
#define _SCFD_UTILS_HIP_TIMER_EVENT_H__

#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include "hip_safe_call.h"
#include "timer_event.h"

namespace scfd
{
namespace utils
{

struct hip_timer_event : public timer_event
{
    hipEvent_t     e_;

    hip_timer_event()
    {
        HIP_SAFE_CALL( hipEventCreate( &e_ ) );
    }
    virtual void    record()
    {
        hipEventRecord( e_, 0 );
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const hip_timer_event *hip_event = dynamic_cast<const hip_timer_event*>(&e0);
        if (hip_event == NULL) {
            throw std::logic_error("hip_timer_event::elapsed_time: try to calc time from different type of timer (non-hip)");
        }
        float   res;
        hipEventSynchronize( e_ );
        hipEventElapsedTime( &res, hip_event->e_, e_ );
        return (double)res;
    };

    virtual ~hip_timer_event()
    {
        hipEventDestroy( e_ );
    }
};

}

}

#endif
