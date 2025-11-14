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


#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>

#include <scfd/memory/cuda.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include "test_custom_index_arranger.h"

int main(int argc, char const *argv[])
{
    using memory_t = scfd::memory::cuda_device;
    using for_each_3_t = scfd::for_each::cuda_nd<3>;
    scfd::utils::init_cuda_persistent();
    auto ret_code = scfd::tests::check<memory_t, for_each_3_t>();
    return ret_code;
}