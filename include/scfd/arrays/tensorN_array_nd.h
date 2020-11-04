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

#ifndef __SCFD_ARRAYS_TENSORN_ARRAY_ND_H__
#define __SCFD_ARRAYS_TENSORN_ARRAY_ND_H__

#include "tensor_array_nd.h"
#include "tensorN_array_nd_view.h"

namespace scfd
{
namespace arrays
{

template<class T, ordinal_type ND, class Memory,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using tensor0_array_nd = tensor_array_nd<T, ND, Memory, Arranger>;
template<class T, ordinal_type ND, class Memory, ordinal_type TensorDim0,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using tensor1_array_nd = tensor_array_nd<T, ND, Memory, Arranger, TensorDim0>;
template<class T, ordinal_type ND, class Memory, ordinal_type TensorDim0, ordinal_type TensorDim1,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using tensor2_array_nd = tensor_array_nd<T, ND, Memory, Arranger, TensorDim0, TensorDim1>;
template<class T, ordinal_type ND, class Memory, ordinal_type TensorDim0, ordinal_type TensorDim1, ordinal_type TensorDim2,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using tensor3_array_nd = tensor_array_nd<T, ND, Memory, Arranger, TensorDim0, TensorDim1, TensorDim2>;
template<class T, ordinal_type ND, class Memory, ordinal_type TensorDim0, ordinal_type TensorDim1, ordinal_type TensorDim2, ordinal_type TensorDim3,
         template <ordinal_type... Dims> class Arranger = detail::default_arranger_chooser<Memory>::template arranger>
using tensor4_array_nd = tensor_array_nd<T, ND, Memory, Arranger, TensorDim0, TensorDim1, TensorDim2, TensorDim3>;

}
}

#endif
