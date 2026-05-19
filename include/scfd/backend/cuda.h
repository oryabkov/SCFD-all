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
//
#ifndef __SCFD_BACKEND_CUDA_H__
#define __SCFD_BACKEND_CUDA_H__


#include <scfd/utils/init_cuda.h>
#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_impl.cuh>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/backend/reduce/thrust.h>
#include <scfd/backend/sort/thrust.h>
#include <scfd/backend/unique/thrust.h>
#include <scfd/backend/exclusive_scan/thrust.h>
#include <scfd/backend/copy/cuda.h>
#include <scfd/backend/inclusive_scan/thrust.h>
#include <scfd/backend/sort_by_key/thrust.h>
#include <scfd/backend/reduce_by_key/thrust.h>
#include <scfd/backend/set_intersection/thrust.h>
#include <scfd/backend/sequence/thrust.h>

namespace scfd
{
namespace backend
{

struct cuda
{
    using memory_type = scfd::memory::cuda_device;
    template <class Ordinal = int>
    using for_each_type = scfd::for_each::cuda<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type      = scfd::for_each::cuda_nd<Dim, Ordinal>;
    using reduce_type           = scfd::thrust_reduce<>;
    using sort_type             = scfd::thrust_sort<>;
    using unique_type           = scfd::thrust_unique<>;
    using exclusive_scan_type   = scfd::thrust_exclusive_scan<>;
    using copy_type             = scfd::cuda_copy<>;
    using inclusive_scan_type   = scfd::thrust_inclusive_scan<>;
    using sort_by_key_type      = scfd::thrust_sort_by_key<>;
    using reduce_by_key_type    = scfd::thrust_reduce_by_key<>;
    using set_intersection_type = scfd::thrust_set_intersection<>;
    using sequence_type         = scfd::thrust_sequence<>;
};

}
}


#endif // __SCFD_BACKEND_CUDA_H__
