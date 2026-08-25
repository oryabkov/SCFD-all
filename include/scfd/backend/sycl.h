// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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
//

#ifndef __SCFD_BACKEND_SYCL_H__
#define __SCFD_BACKEND_SYCL_H__

#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/sycl_common.h>

namespace scfd
{
namespace backend
{
struct sycl : public sycl_common
{
    using runtime_type = sycl;

    template <class Log>
    static int init_device( Log &, int device_id = 0 )
    {
        return init_device( device_id );
    }

    static int init_device( int device_id = 0 )
    {
        std::vector<::sycl::device> devices = ::sycl::device::get_devices( ::sycl::info::device_type::gpu );
        if ( devices.empty() )
            throw std::runtime_error( "sycl::init_device: no visible SYCL GPU devices" );
        if ( device_id < 0 || device_id >= static_cast<int>( devices.size() ) )
            throw std::runtime_error(
                "sycl::init_device: requested device " + std::to_string( device_id ) +
                " is outside visible SYCL GPU device range"
            );
        sycl_device_queue = ::sycl::queue( devices[static_cast<std::size_t>( device_id )] );
        return device_id;
    }
};
}
}

#endif // __SCFD_BACKEND_SYCL_H__
