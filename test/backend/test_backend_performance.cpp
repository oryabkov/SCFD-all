#include "test_backend_config.h"
#include <scfd/backend/backend.h>

#include "test_backend_performance_common.h"

int main()
{
    return scfd_backend_tests::run_backend_performance_tests<scfd_backend_tests::expected_backend>(
        scfd_backend_tests::expected_backend_configuration_name(),
        scfd_backend_tests::expected_backend_requires_acceleration
    );
}
