// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_BACKEND_COMMON_H__
#define __SCFD_BACKEND_COMMON_H__

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace scfd
{
namespace backend
{
namespace detail
{

struct device_memory_info
{
    std::size_t free_bytes;
    std::size_t total_bytes;
    bool        free_bytes_known;
    bool        total_bytes_known;

    device_memory_info() : free_bytes( 0 ), total_bytes( 0 ), free_bytes_known( false ), total_bytes_known( false )
    {
    }

    device_memory_info( std::size_t free_bytes_, std::size_t total_bytes_, bool free_known_, bool total_known_ )
        : free_bytes( free_bytes_ ), total_bytes( total_bytes_ ), free_bytes_known( free_known_ ),
          total_bytes_known( total_known_ )
    {
    }
};

struct host_memory_info
{
    std::uint64_t rss_bytes;
    std::uint64_t peak_rss_bytes;
    std::uint64_t virtual_bytes;
    std::uint64_t swap_bytes;
    std::uint64_t system_total_bytes;
    std::uint64_t system_free_bytes;
    std::uint64_t system_available_bytes;
    bool          process_memory_known;
    bool          peak_rss_known;
    bool          virtual_memory_known;
    bool          swap_memory_known;
    bool          system_memory_known;

    host_memory_info()
        : rss_bytes( 0 ), peak_rss_bytes( 0 ), virtual_bytes( 0 ), swap_bytes( 0 ), system_total_bytes( 0 ),
          system_free_bytes( 0 ), system_available_bytes( 0 ), process_memory_known( false ), peak_rss_known( false ),
          virtual_memory_known( false ), swap_memory_known( false ), system_memory_known( false )
    {
    }
};

inline bool read_status_kb_field( const std::string &key, std::uint64_t &value_bytes )
{
#if defined( __linux__ )
    std::ifstream status( "/proc/self/status" );
    std::string   line;
    while ( std::getline( status, line ) )
    {
        if ( line.compare( 0, key.size(), key ) == 0 )
        {
            std::istringstream value_stream( line.substr( key.size() ) );
            std::uint64_t      value_kb = 0;
            value_stream >> value_kb;
            if ( value_stream.fail() )
                return false;
            value_bytes = value_kb * 1024ull;
            return true;
        }
    }
#else
    (void)key;
    (void)value_bytes;
#endif
    return false;
}

inline void read_system_meminfo( host_memory_info &info )
{
#if defined( __linux__ )
    std::ifstream meminfo( "/proc/meminfo" );
    std::string   line;
    while ( std::getline( meminfo, line ) )
    {
        const char *fields[] = { "MemTotal:", "MemFree:", "MemAvailable:" };
        for ( int i = 0; i < 3; ++i )
        {
            const std::string key( fields[i] );
            if ( line.compare( 0, key.size(), key ) == 0 )
            {
                std::istringstream value_stream( line.substr( key.size() ) );
                std::uint64_t      value_kb = 0;
                value_stream >> value_kb;
                if ( value_stream.fail() )
                    continue;
                if ( i == 0 )
                    info.system_total_bytes = value_kb * 1024ull;
                else if ( i == 1 )
                    info.system_free_bytes = value_kb * 1024ull;
                else
                    info.system_available_bytes = value_kb * 1024ull;
                info.system_memory_known = true;
            }
        }
    }
#else
    (void)info;
#endif
}

inline host_memory_info get_host_memory_info()
{
    host_memory_info info;
    std::uint64_t    value = 0;
    if ( read_status_kb_field( "VmRSS:", value ) )
    {
        info.rss_bytes            = value;
        info.process_memory_known = true;
    }
    if ( read_status_kb_field( "VmHWM:", value ) )
    {
        info.peak_rss_bytes = value;
        info.peak_rss_known = true;
    }
    if ( read_status_kb_field( "VmSize:", value ) )
    {
        info.virtual_bytes        = value;
        info.virtual_memory_known = true;
    }
    if ( read_status_kb_field( "VmSwap:", value ) )
    {
        info.swap_bytes        = value;
        info.swap_memory_known = true;
    }
    read_system_meminfo( info );
    return info;
}

}
}
}

#endif
