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

#ifndef __SCFD_TRIVIAL_PLATFORM_H__
#define __SCFD_TRIVIAL_PLATFORM_H__

#include <memory>
#include "trivial_comm.h"

namespace scfd
{
namespace communication
{

/// Analog of mpi_wrap, but WITHOUT any MPI. It owns the single shared message queue
/// and hands out trivial_comm handles onto it. Deliberately NON-copyable and
/// NON-movable so the queue keeps one stable address shared by every handle.
template <class Memory>
struct trivial_platform
{
    using queue_type = detail::trivial_message_queue<Memory>;
    using comm_type  = trivial_comm<Memory>;

    /// argc/argv kept only for signature parity with mpi_wrap; no MPI_Init here.
    trivial_platform( int argc, char *argv[] )
    {
        (void)argc;
        (void)argv;
        queue_ = std::make_unique<queue_type>();
    }
    ~trivial_platform() = default;

    trivial_platform( const trivial_platform & )            = delete;
    trivial_platform &operator=( const trivial_platform & ) = delete;
    trivial_platform( trivial_platform && )                 = delete;
    trivial_platform &operator=( trivial_platform && )      = delete;

    /// Mirrors mpi_wrap::comm_world(): a num_procs==1, myid==0 handle onto the queue.
    comm_type comm_world() const
    {
        return comm_type( 1, 0, queue_.get() );
    }

private:
    std::unique_ptr<queue_type> queue_;
};

} // namespace communication
} // namespace scfd

#endif
