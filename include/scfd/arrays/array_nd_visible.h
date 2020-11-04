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

#ifndef __SCFD_ARRAYS_ARRAY_ND_VISIBLE_H__
#define __SCFD_ARRAYS_ARRAY_ND_VISIBLE_H__

#include "tensor_array_nd_visible.h"

namespace scfd
{
namespace arrays
{

template<class T, ordinal_type ND, class Memory,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using array_nd_visible = tensor_array_nd_visible<T, ND, Memory, Arranger>;

}
}

#endif
