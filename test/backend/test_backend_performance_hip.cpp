#include <scfd/backend/hip.h>

#define PLATFORM_HIP
#include <scfd/backend/backend.h>

#include "test_backend_performance_common.h"

int main()
{
    return scfd_backend_tests::run_backend_performance_tests<scfd::backend::hip>( "hip", true );
}
