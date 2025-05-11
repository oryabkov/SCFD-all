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

#ifndef __SCFD_BACKEND_H__
#define __SCFD_BACKEND_H__


#if   defined(PLATFORM_SERIAL_CPU)

#include <scfd/memory/host.h>
#include <scfd/for_each/serial_cpu.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/reduce/serial_cpu.h>


#elif defined(PLATFORM_OMP)

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_impl.h>
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/reduce/omp_reduce_impl.h>


#elif defined(PLATFORM_CUDA)

#include <scfd/utils/init_cuda.h>
#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_impl.cuh>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/reduce/thrust.h>


#elif defined(PLATFORM_HIP)

#include <scfd/utils/init_hip.h>
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_impl.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>


#elif defined(PLATFORM_SYCL)

#include <scfd/memory/sycl.h>
#include <scfd/for_each/sycl_impl.h>
#include <scfd/for_each/sycl_nd_impl.h>
#include <scfd/reduce/sycl_reduce_impl.h>


#define MAKE_SYCL_DEVICE_COPYABLE(kernel) template<>    \
struct sycl::is_device_copyable<typename kernel>        \
    : std::true_type {}


#else
#error "No platform has been chosen for backend"

#endif

namespace scfd
{

struct backend
{
#if   defined(PLATFORM_SERIAL_CPU)
    using memory_type       = scfd::memory::host;
    template <class ordinal = int>
    using for_each_type     = scfd::for_each::serial_cpu<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd_type  = scfd::for_each::serial_cpu_nd<dim, ordinal>;
    using reduce_type       = scfd::serial_cpu_reduce<>;


#elif defined(PLATFORM_OMP)
    using memory_type       = scfd::memory::host;
    template <class ordinal = int>
    using for_each_type     = scfd::for_each::omp<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd_type  = scfd::for_each::omp_nd<dim, ordinal>;
    using reduce_type       = scfd::omp_reduce<>;


#elif defined(PLATFORM_CUDA)
    using memory_type       = scfd::memory::cuda_device;
    template <class ordinal = int>
    using for_each_type     = scfd::for_each::cuda<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd_type  = scfd::for_each::cuda_nd<dim, ordinal>;
    using reduce_type       = scfd::thrust_reduce<>;


#elif defined(PLATFORM_HIP)
    using memory_type       = scfd::memory::hip_device;
    template <class ordinal = int>
    using for_each_type     = scfd::for_each::hip<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd_type  = scfd::for_each::hip_nd<dim, ordinal>;
    using reduce_type       = scfd::thrust_reduce<>;


#elif defined(PLATFORM_SYCL)
    using memory_type       = scfd::memory::sycl_device;
    template <class ordinal = int>
    using for_each_type     = scfd::for_each::sycl<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd_type  = scfd::for_each::sycl_nd<dim, ordinal>;
    using reduce_type       = scfd::sycl_reduce<>;


#endif

    // usefull aliases
    using memory      = memory_type;
    template <class ordinal = int>
    using for_each    = for_each_type<ordinal>;
    template <int dim, class ordinal = int>
    using for_each_nd = for_each_nd_type<dim, ordinal>;
    using reduce      = reduce_type;
};

} //scfd

#endif
