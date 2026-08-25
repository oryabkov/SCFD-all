#include <iostream>
#include <type_traits>

#include "test_backend_config.h"
#include <scfd/backend/backend.h>

int main()
{
    using backend_t     = scfd_backend_tests::expected_backend;
    using backend_def_t = scfd::backend::current;

    using memory_t      = scfd::backend::memory;
    using for_each_t    = scfd::backend::for_each<int>;
    using for_each_nd_t = scfd::backend::for_each_nd<3>;
    using reduce_t      = scfd::backend::reduce;

    if ( !std::is_same<backend_t, backend_def_t>::value )
    {
        std::cout << "FAILED current backend type check" << std::endl;
        return 10;
    }
    if ( !std::is_same<typename backend_t::memory_type, memory_t>::value )
    {
        std::cout << "FAILED memory type check" << std::endl;
        return 11;
    }
    if ( !std::is_same<typename backend_t::template for_each_type<int>, for_each_t>::value )
    {
        std::cout << "FAILED for_each type check" << std::endl;
        return 12;
    }
    if ( !std::is_same<typename backend_t::template for_each_nd_type<3>, for_each_nd_t>::value )
    {
        std::cout << "FAILED for_each_nd type check" << std::endl;
        return 13;
    }
    if ( !std::is_same<typename backend_t::reduce_type, reduce_t>::value )
    {
        std::cout << "FAILED reduce type check" << std::endl;
        return 14;
    }

    std::cout << scfd_backend_tests::expected_backend_configuration_name() << ": PASSED" << std::endl;
    return 0;
}
