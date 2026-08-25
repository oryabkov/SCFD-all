#include "test_backend_config.h"
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

int main()
{
    return scfd_backend_tests::run_backend_algorithm_tests<scfd_backend_tests::expected_backend>(
        scfd_backend_tests::expected_backend_configuration_name()
    );
}
