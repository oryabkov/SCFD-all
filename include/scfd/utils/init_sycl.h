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

#ifndef __SCFD_INIT_SYCL_
#define __SCFD_INIT_SYCL_

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

namespace dpl = oneapi::dpl;

// TODO: singleton impl, gpu/cpu selection
inline sycl::queue sycl_device_queue(sycl::gpu_selector_v);

#endif
