
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/reduce/omp_reduce_impl.h>
#include "poisson_solver_impl.h"
#include "current_poisson_solver.h"

template class poisson_solver<real,for_each_t,reduce_t,memory_t>;

