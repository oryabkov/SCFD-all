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

#ifndef __SCFD_UTILS_DEVICE_TAG_H__
#define __SCFD_UTILS_DEVICE_TAG_H__

#if defined(__CUDACC__) || defined(__HIPCC__)
#define __DEVICE_TAG__ __device__ __host__
#else
#define __DEVICE_TAG__
#endif

/*#ifndef __CUDACC__
#define __DEVICE_ONLY_TAG__
#else
#define __DEVICE_ONLY_TAG__ __device__
#endif*/

#endif
