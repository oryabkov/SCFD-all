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

#ifndef __SCFD_SYCL_SORT_BY_KEY_H__
#define __SCFD_SYCL_SORT_BY_KEY_H__

#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct sycl_sort_by_key
{
    template <class Key, class Value, class Compare>
    void operator()( Ord size, Key *keys, Value *values, Compare compare ) const;

    template <class Key, class Value>
    void operator()( Ord size, Key *keys, Value *values ) const
    {
        operator()( size, keys, values, functional::less<Key>() );
    }

    void wait() const
    {
    }
};

}

#endif
