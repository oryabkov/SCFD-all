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

#ifndef __SCFD_BACKEND_OMP_H__
#define __SCFD_BACKEND_OMP_H__

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_impl.h>
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/backend/reduce/omp_reduce_impl.h>
#include <scfd/backend/sort/omp_sort_impl.h>
#include <scfd/backend/unique/omp_unique_impl.h>
#include <scfd/backend/exclusive_scan/omp_exclusive_scan_impl.h>
#include <scfd/backend/copy/omp_copy_impl.h>
#include <scfd/backend/inclusive_scan/omp_inclusive_scan_impl.h>
#include <scfd/backend/sort_by_key/omp_sort_by_key_impl.h>
#include <scfd/backend/reduce_by_key/omp_reduce_by_key_impl.h>
#include <scfd/backend/set_intersection/omp_set_intersection_impl.h>
#include <scfd/backend/sequence/omp_sequence_impl.h>
#include <scfd/backend/count_by_key/omp_count_by_key_impl.h>
#include <scfd/backend/runtime/omp.h>


namespace scfd
{
namespace backend
{

struct omp
{
    using memory_type = scfd::memory::host;
    template <class Ordinal = int>
    using for_each_type = scfd::for_each::openmp<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type      = scfd::for_each::openmp_nd<Dim, Ordinal>;
    using reduce_type           = scfd::omp_reduce<>;
    using sort_type             = scfd::omp_sort<>;
    using unique_type           = scfd::omp_unique<>;
    using exclusive_scan_type   = scfd::omp_exclusive_scan<>;
    using copy_type             = scfd::omp_copy<>;
    using inclusive_scan_type   = scfd::omp_inclusive_scan<>;
    using sort_by_key_type      = scfd::omp_sort_by_key<>;
    using reduce_by_key_type    = scfd::omp_reduce_by_key<>;
    using set_intersection_type = scfd::omp_set_intersection<>;
    using sequence_type         = scfd::omp_sequence<>;
    using count_by_key_type     = scfd::omp_count_by_key<>;
    using runtime_type          = scfd::backend::detail::omp_runtime;
};

}
}

#endif // __SCFD_BACKEND_OMP_H__
