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

#ifndef __SCFD_SYCL_SORT_BY_KEY_IMPL_H__
#define __SCFD_SYCL_SORT_BY_KEY_IMPL_H__

#include "sycl_sort_by_key.h"
#include <scfd/backend/value_pair.h>
#include <scfd/utils/init_sycl.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

namespace scfd
{
namespace detail
{

template <class Pair, class Compare>
struct sycl_value_pair_less
{
    sycl_value_pair_less( Compare compare_ ) : compare( compare_ )
    {
    }

    bool operator()( const Pair &a, const Pair &b ) const
    {
        return compare( a.first, b.first );
    }

    Compare compare;
};

}

template <class Ord>
template <class Key, class Value, class Compare>
void sycl_sort_by_key<Ord>::operator()( Ord size, Key *keys, Value *values, Compare compare ) const
{
    if ( size <= 0 )
        return;

    typedef scfd::backend::value_pair<Key, Value> pair_type;
    pair_type *pairs = sycl::malloc_device<pair_type>( static_cast<size_t>( size ), sycl_device_queue );

    sycl_device_queue
        .parallel_for(
            sycl::range<1>( static_cast<size_t>( size ) ),
            [=]( sycl::id<1> item ) {
                const Ord i     = static_cast<Ord>( item[0] );
                pairs[i].first  = keys[i];
                pairs[i].second = values[i];
            }
        )
        .wait();

    auto policy = dpl::execution::make_device_policy( sycl_device_queue );
    dpl::sort( policy, pairs, pairs + size, detail::sycl_value_pair_less<pair_type, Compare>( compare ) );

    sycl_device_queue
        .parallel_for(
            sycl::range<1>( static_cast<size_t>( size ) ),
            [=]( sycl::id<1> item ) {
                const Ord i = static_cast<Ord>( item[0] );
                keys[i]     = pairs[i].first;
                values[i]   = pairs[i].second;
            }
        )
        .wait();

    sycl::free( pairs, sycl_device_queue );
}

}

#endif
