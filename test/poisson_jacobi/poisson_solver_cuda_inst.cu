
#include <scfd/for_each/cuda_nd_impl.cuh>
#include "poisson_solver_impl.h"
#include "current_poisson_solver.h"

template class poisson_solver<real,for_each_t,reduce_t,memory_t>;

