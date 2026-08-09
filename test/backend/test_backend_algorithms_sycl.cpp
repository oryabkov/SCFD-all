#include <scfd/backend/sycl.h>

#define PLATFORM_SYCL
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd::backend::sycl>( "sycl" );
}
