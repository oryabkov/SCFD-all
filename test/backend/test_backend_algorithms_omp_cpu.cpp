#include <scfd/backend/omp.h>

#define PLATFORM_OMP
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd::backend::omp>( "omp" );
}
