#ifndef __SCFD_TEST_FOR_PAPER_TEST_CHECKS_H__
#define __SCFD_TEST_FOR_PAPER_TEST_CHECKS_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace test_checks
{
inline bool parse_positive( const char *text, std::size_t &value )
{
    const std::string input( text );
    if ( input.empty() || input.find_first_not_of( "0123456789" ) != std::string::npos )
        return false;
    try
    {
        const auto parsed = std::stoull( input );
        if ( parsed == 0 || parsed > static_cast<unsigned long long>( std::numeric_limits<int>::max() / 3 ) )
            return false;
        value = static_cast<std::size_t>( parsed );
        return true;
    }
    catch ( const std::exception & )
    {
        return false;
    }
}

template <class T>
T difference( T expected, T actual )
{
    if ( !std::isfinite( expected ) || !std::isfinite( actual ) )
        return std::numeric_limits<T>::infinity();
    return std::abs( expected - actual );
}

template <class T>
bool check( const char *name, T maximum_difference )
{
    // Inputs are in [-100,100]. Allow floating-point multiply/subtract and FMA
    // rounding, while checking every component (not an average over the array).
    const T    tolerance = std::numeric_limits<T>::epsilon() * T( 10000 ) * T( 16 );
    const bool passed    = std::isfinite( maximum_difference ) && maximum_difference <= tolerance;
    std::cout << name << " maximum difference = " << maximum_difference << ( passed ? " PASS" : " FAIL" ) << std::endl;
    return passed;
}
}

#endif
