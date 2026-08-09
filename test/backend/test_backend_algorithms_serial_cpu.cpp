#include <scfd/backend/serial_cpu.h>

#define PLATFORM_SERIAL_CPU
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd::backend::serial_cpu>( "serial_cpu" );
}
