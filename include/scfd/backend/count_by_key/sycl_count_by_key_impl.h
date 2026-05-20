// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_SYCL_COUNT_BY_KEY_IMPL_H__
#define __SCFD_SYCL_COUNT_BY_KEY_IMPL_H__

#include "sycl_count_by_key.h"
#include <scfd/backend/count_by_key/serial_cpu.h>
#include <scfd/utils/init_sycl.h>
#include <vector>

namespace scfd
{

template <class Ord>
template <class Key, class Count, class KeyEqual>
Ord sycl_count_by_key<Ord>::operator()(
    Ord size, const Key *keys_in, Key *keys_out, Count *counts_out, KeyEqual key_equal
) const
{
    if ( size <= 0 )
        return 0;

    std::vector<Key>   keys_in_host( static_cast<std::size_t>( size ) );
    std::vector<Key>   keys_out_host( static_cast<std::size_t>( size ) );
    std::vector<Count> counts_out_host( static_cast<std::size_t>( size ) );

    sycl_device_queue.memcpy( keys_in_host.data(), keys_in, sizeof( Key ) * static_cast<std::size_t>( size ) ).wait();

    const Ord out_size = detail::count_by_key_host_impl(
        size, keys_in_host.data(), keys_out_host.data(), counts_out_host.data(), key_equal
    );

    sycl_device_queue.memcpy( keys_out, keys_out_host.data(), sizeof( Key ) * static_cast<std::size_t>( out_size ) )
        .wait();
    sycl_device_queue
        .memcpy( counts_out, counts_out_host.data(), sizeof( Count ) * static_cast<std::size_t>( out_size ) )
        .wait();

    return out_size;
}

}

#endif
