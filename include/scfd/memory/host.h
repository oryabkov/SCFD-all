// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_MEMORY_HOST_H__
#define __SCFD_MEMORY_HOST_H__

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace scfd
{
namespace memory
{

//common interface
//TODO make MemoryType concept

struct host
{
    typedef     host            host_memory_type;
    typedef     void*           pointer_type;
    typedef     const void*     const_pointer_type;

    static const bool           is_host_visible = true;
    static const bool           prefer_array_of_structs = true;

    static void    malloc(pointer_type* p, size_t size)
    {
        if (size != 0)
        { 
            *p = std::malloc(size); 
            if (*p == NULL) throw std::runtime_error("host::malloc: malloc failed");
        }
        else
        {
            *p = NULL;
        } 
    }
    static void    free(pointer_type p)
    {
        std::free(p);
    }

    static void    copy_to_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        //TODO error handling (how?)
        std::memcpy( dst, src, size );
    }
    static void    copy_from_host(size_t size, const_pointer_type src, pointer_type dst)
    {
        //TODO error handling (how?)
        std::memcpy( dst, src, size );
    }
};

}

}

#endif
