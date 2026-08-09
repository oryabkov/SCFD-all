#include <scfd/backend/cuda.h>

#define PLATFORM_CUDA
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd::backend::cuda>( "cuda" );
}
