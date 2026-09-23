#ifndef __SCFD_TEST_MATMUL_ARGUMENTS_H__
#define __SCFD_TEST_MATMUL_ARGUMENTS_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

inline bool parse_benchmark_integer( const char *text, std::size_t &value )
{
    const std::string input( text );
    if ( input.empty() || input.find_first_not_of( "0123456789" ) != std::string::npos )
        return false;
    try
    {
        const auto parsed = std::stoull( input );
        if ( parsed > static_cast<unsigned long long>( std::numeric_limits<int>::max() ) )
            return false;
        value = static_cast<std::size_t>( parsed );
        return true;
    }
    catch ( const std::exception & )
    {
        return false;
    }
}

#endif
