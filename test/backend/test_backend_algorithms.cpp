#include <string>

#include "test_backend_config.h"
#include <scfd/backend/backend.h>

#include "test_backend_algorithms_common.h"

static_assert(
    std::is_same<scfd::backend::current<>, scfd_backend_tests::expected_backend<>>::value,
    "current uses the configured default ordinal"
);

int main()
{
    return scfd_backend_tests::run_backend_ordinal_tests( []( auto backend, const char *ordinal_name ) {
        using backend_t = typename decltype( backend )::type;
        const std::string name =
            std::string( scfd_backend_tests::expected_backend_configuration_name() ) + "/" + ordinal_name;
        return scfd_backend_tests::run_backend_algorithm_tests<backend_t>( name.c_str() );
    } );
}
