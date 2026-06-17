// Copyright © 2016 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_TRIVIAL_COMM_H__
#define __SCFD_TRIVIAL_COMM_H__

#include <cstddef>
#include <stdexcept>
#include "trivial_message_queue.h"

namespace scfd
{
namespace communication
{
namespace detail
{

/// Analog of detail::mpi_request. In the synchronous trivial impl it only has to
/// carry enough info for waitany to report which message just "completed".
struct trivial_request
{
    int  source   = 0;
    int  tag      = 0;
    bool reported = false;  // waitany flips this once it has returned this request's index
};

/// Analog of detail::mpi_status. rect_distributor reads source()/tag() after waitany
/// to find which packet/bucket the completed message belongs to.
struct trivial_status
{
    int source() const { return source_; }
    int tag() const { return tag_; }

    int source_ = 0;
    int tag_    = 0;
};

} // namespace detail

/// Single-rank stand-in for mpi_comm_info (num_procs == 1, myid == 0).
/// A lightweight COPYABLE handle holding a non-owning pointer to the shared queue
/// (the queue itself is owned by trivial_platform). This is the type plugged into
/// rect_partitioner / rect_distributor as the Comm template parameter.
template <class Memory>
struct trivial_comm
{
    using queue_type   = detail::trivial_message_queue<Memory>;
    using request_type = detail::trivial_request;
    using status_type  = detail::trivial_status;

    int         num_procs = 1;
    int         myid      = 0;
    queue_type *queue     = nullptr;  // non-owning, points at trivial_platform's queue

    trivial_comm() = default;
    trivial_comm( int num_procs_p, int myid_p, queue_type *queue_p )
        : num_procs( num_procs_p ), myid( myid_p ), queue( queue_p )
    {
    }

    /// Park buf into the queue as message (myid -> dest, tag). No copy is made here.
    template <class T>
    void isend( const T *buf, int count, int dest, int tag, request_type &request ) const
    {
        // TODO: queue->push( myid, dest, tag, buf, count * sizeof(T) );
        //       optionally stamp dest/tag into request.
        queue->push( myid, dest, tag, buf, count * sizeof(T) );
    }

    /// Synchronously receive message (source -> myid, tag) into buf (copy happens here).
    template <class T>
    void irecv( T *buf, int count, int source, int tag, request_type &request ) const
    {
        // TODO: queue->recv( source, myid, tag, buf, count * sizeof(T) );
        //       record source/tag into request so waitany can report them.
        queue->recv( source, myid, tag, buf, count * sizeof(T) );
        request.source = source;
        request.tag = tag;
        request.reported = false;
    }

    /// Return the index of the next not-yet-reported request and fill *status with its
    /// source/tag. All transfers already happened, so just walk requests in order.
    int waitany( int count, request_type *requests, status_type *status ) const
    {
        // TODO: find first requests[i] with reported == false; set it true;
        //       status->source_ = requests[i].source; status->tag_ = requests[i].tag;
        //       return i.
        for ( int i = 0; i < count; i++ )
        {
            if ( requests[i].reported == false )
            {
                requests[i].reported = true;
                status->source_ = requests[i].source;
                status->tag_ = requests[i].tag;

                return i;
            }
        }

        throw std::logic_error( "trivial_comm::waitany: no un-reported request left" );
    }

    /// isend requests need no real waiting in the synchronous trivial impl.
    void waitall( int count, request_type *requests ) const
    {
        // TODO: nothing to wait for (messages were consumed synchronously by irecv).
    }
};

} // namespace communication
} // namespace scfd

#endif
