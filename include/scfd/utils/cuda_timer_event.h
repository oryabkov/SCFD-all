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

#ifndef _SCFD_UTILS_CUDA_TIMER_EVENT_H__
#define _SCFD_UTILS_CUDA_TIMER_EVENT_H__

#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cuda_safe_call.h"
#include "timer_event.h"

namespace scfd
{
namespace utils
{

struct cuda_timer_event : public timer_event
{
    cudaEvent_t     e_;

    cuda_timer_event()
    {
        CUDA_SAFE_CALL( cudaEventCreate( &e_ ) );
    }
    virtual void    record()
    {
        cudaEventRecord( e_, 0 );
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const cuda_timer_event *cuda_event = dynamic_cast<const cuda_timer_event*>(&e0);
        if (cuda_event == NULL) {
            throw std::logic_error("cuda_timer_event::elapsed_time: try to calc time from different type of timer (non-cuda)");
        }
        float   res;
        cudaEventSynchronize( e_ );
        cudaEventElapsedTime( &res, cuda_event->e_, e_ );
        return (double)res;
    };

    virtual ~cuda_timer_event()
    {
        cudaEventDestroy( e_ );
    }
};

}

}

#endif
