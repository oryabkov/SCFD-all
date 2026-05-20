#include <scfd/backend/omp.h>

#define PLATFORM_OMP
#include <scfd/backend/backend.h>

#include "test_backend_runtime_common.h"

int main()
{
    return scfd_backend_tests::run_backend_runtime_tests<scfd::backend::omp>( "omp_runtime" );
}
