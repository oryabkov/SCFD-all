#include <scfd/for_each/hip_nd_impl.h>

#ifdef POISSON_SOLVER_USE_LAMBDA
#include "poisson_solver_lambda_impl.h"
#else
#include "poisson_solver_impl.h"
#endif
#include "current_poisson_solver.h"

template class poisson_solver<real,for_each_t,reduce_t,memory_t>;

