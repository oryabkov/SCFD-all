#include <scfd/backend/hip.h>

#define PLATFORM_HIP
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd::backend::hip>( "hip" );
}
