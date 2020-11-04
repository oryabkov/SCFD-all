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

#ifndef __SCFD_UTILS_CUDA_STREAM_WRAP_H__
#define __SCFD_UTILS_CUDA_STREAM_WRAP_H__

#include <cuda_runtime.h>

namespace scfd
{
namespace utils
{

class cuda_stream_wrap
{
public:
    cuda_stream_wrap(bool do_init = false) 
    {
        is_inited_ = false;
        if (do_init) init();
    }
    cuda_stream_wrap(const cuda_stream_wrap &s) = delete;
    cuda_stream_wrap(cuda_stream_wrap &&s)
    {
        is_inited_ = s.is_inited_;
        if (s.is_inited_) stream_ = s.stream_;
        s.is_inited_ = false;
    }

    cuda_stream_wrap &opeartor=(const cuda_stream_wrap &s) = delete;
    cuda_stream_wrap &opeartor=(cuda_stream_wrap &&s)
    {
        free();

        /// TODO a little code duplication with constructor - can create distinct method move
        is_inited_ = s.is_inited_;
        if (s.is_inited_) stream_ = s.stream_;
        s.is_inited_ = false;   

        return *this;
    }

    ~cuda_stream_wrap()
    {
        free();
    }

    bool            is_inited()const { return is_inited_; }
    cudaStream_t    stream()const { assert(is_inited_); return stream_; }

    void            init()
    {
        assert(!is_inited_);
        CUDA_SAFE_CALL( cudaStreamCreate(stream_) );
        is_inited_ = true;
    }
    /// NOTE exception here means in fact logic error
    void            free()
    {
        if (!is_inited_) return;
        is_inited_ = false;
        CUDA_SAFE_CALL( cudaStreamDestroy(stream_) );
    }

private:
    bool            is_inited_;
    /// stream_ only has meaning if is_inited_==true
    cudaStream_t    stream_;
};

}
}

#endif
