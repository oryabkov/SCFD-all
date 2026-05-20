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

#ifndef __SCFD_OMP_COUNT_BY_KEY_H__
#define __SCFD_OMP_COUNT_BY_KEY_H__

#include <scfd/backend/functional/basic_ops.h>

namespace scfd
{

template <class Ord = int>
struct omp_count_by_key
{
    template <class Key, class Count, class KeyEqual>
    Ord operator()( Ord size, const Key *keys_in, Key *keys_out, Count *counts_out, KeyEqual key_equal ) const;

    template <class Key, class Count>
    Ord operator()( Ord size, const Key *keys_in, Key *keys_out, Count *counts_out ) const
    {
        return operator()( size, keys_in, keys_out, counts_out, functional::equal_to<Key>() );
    }

    void wait() const
    {
    }
};

}

#endif
