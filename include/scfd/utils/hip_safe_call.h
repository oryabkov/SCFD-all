// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Ivan Antonovich Sorokin

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

#ifndef __SCFD_UTILS_HIP_SAFE_CALL_H__
#define __SCFD_UTILS_HIP_SAFE_CALL_H__

#include <stdexcept>
#include <string>
#include <hip/hip_runtime.h>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define HIP_SAFE_CALL(X)                                                                                                                                                                       \
    do {                                                                                                                                                                                        \
        hipError_t hip_res = (X);                                                                                                                                                             \
        if (hip_res != hipSuccess) throw std::runtime_error(std::string("HIP_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") + std::string(hipGetErrorString(hip_res)));    \
    } while (0)

#endif
