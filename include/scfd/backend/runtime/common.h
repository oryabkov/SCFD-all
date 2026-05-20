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

#ifndef __SCFD_BACKEND_RUNTIME_COMMON_H__
#define __SCFD_BACKEND_RUNTIME_COMMON_H__

#include <cstddef>

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

}
}
}

#endif
