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

#ifndef __SCFD_ARRAYS_DEFAULT_ARRANGER_CHOOSER_H__
#define __SCFD_ARRAYS_DEFAULT_ARRANGER_CHOOSER_H__

#include "../last_index_fast_arranger.h"
#include "../first_index_fast_arranger.h"

namespace scfd
{
namespace arrays
{
namespace detail
{

template<bool ArrayOfStructs>
struct default_arranger_chooser_
{
    //template<ordinal_type... Dims>
    //using arranger = last_index_fast_arranger<Ord,Dims...>;
};

template<>
struct default_arranger_chooser_<true>
{
    template<ordinal_type... Dims>
    using arranger = last_index_fast_arranger<Dims...>;
};

template<>
struct default_arranger_chooser_<false>
{
    template<ordinal_type... Dims>
    using arranger = first_index_fast_arranger<Dims...>;
};

template<class Memory>
struct default_arranger_chooser
{
    template<ordinal_type... Dims>
    using arranger = typename default_arranger_chooser_<Memory::prefer_array_of_structs>::template arranger<Dims...>;
    //using arranger = default_arranger_chooser_<true>::arranger_<Ord,Dims...>;
};

}
}
}

#endif
