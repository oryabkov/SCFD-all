// Copyright © 2023-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_MPI_RECT_DISTRIBUTOR_H__
#define __SCFD_MPI_RECT_DISTRIBUTOR_H__

#include "mpi_comm_info.h"
#include "rect_distributor.h"

namespace scfd
{
namespace communication
{

template <
    class T, int Dim, class Memory, class ForEach, class Ord, class BigOrd, class Comm = mpi_comm_info>
struct mpi_rect_distributor : rect_distributor<T, Dim, Memory, ForEach, Ord, BigOrd, Comm>
{
    using base_t = rect_distributor<T, Dim, Memory, ForEach, Ord, BigOrd, Comm>;

    // using base_t::base_t;
};

} // namespace communication
} // namespace scfd

#endif
