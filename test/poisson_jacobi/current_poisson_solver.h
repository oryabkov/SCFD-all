#ifndef __CURRENT_POISSON_SOLVER_H__
#define __CURRENT_POISSON_SOLVER_H__

#include "poisson_solver.h"
#include "current_dim.h"

#if defined(POISSON_SOLVER_SERIAL)

#include <scfd/memory/host.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/reduce/serial_cpu.h>
///NOTE for serial implementation no separate intantiation is needed
#ifdef POISSON_SOLVER_USE_LAMBDA
#include "poisson_solver_lambda_impl.h"
#else
#include "poisson_solver_impl.h"
#endif

using memory_t = scfd::memory::host;
using for_each_t = scfd::for_each::serial_cpu_nd<current_dim>;
using reduce_t = scfd::serial_cpu_reduce<>;

#elif defined(POISSON_SOLVER_OMP)

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_nd.h>
#include <scfd/reduce/omp_reduce.h>

using memory_t = scfd::memory::host;
using for_each_t = scfd::for_each::openmp_nd<current_dim>;
using reduce_t = scfd::omp_reduce<>;

#elif defined(POISSON_SOLVER_CUDA)

#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/reduce/thrust.h>

using memory_t = scfd::memory::cuda_device;
using for_each_t = scfd::for_each::cuda_nd<current_dim>;
using reduce_t = scfd::thrust_reduce<>;

#else

#error "None of POISSON_SOLVER_ macro is defined!"

#endif

using real = float;
using poisson_solver_t = poisson_solver<real,for_each_t,reduce_t,memory_t,current_dim>;

#endif
