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

#ifndef __SCFD_TRIVIAL_MESSAGE_QUEUE_H__
#define __SCFD_TRIVIAL_MESSAGE_QUEUE_H__

#include <map>
#include <cstddef>
#include <cassert>
#include <stdexcept>

namespace scfd
{
namespace communication
{
namespace detail
{

/// In-process "transport" for trivial_comm: the place where
/// data lives between an isend and the matching irecv.
template <class Memory>
class trivial_message_queue
{
public:
    using mem_t = Memory;

    /// who -> whom + message tag. For the single-rank case from/to are always 0.
    struct key_type
    {
        int from;
        int to;
        int tag;

        bool operator<( const key_type &o ) const
        {
            if ( from != o.from )
                return from < o.from;
            if ( to != o.to )
                return to < o.to;
            return tag < o.tag;
        }
    };

    /// Non-owning view onto the sender buffer.
    struct entry_type
    {
        const void *data; // pointer to the sender buffer
        int         size; // size in bytes
    };

    trivial_message_queue() = default;

    trivial_message_queue( const trivial_message_queue & )            = delete;
    trivial_message_queue &operator=( const trivial_message_queue & ) = delete;
    trivial_message_queue( trivial_message_queue && )                 = default;
    trivial_message_queue &operator=( trivial_message_queue && )      = default;

    /// Called by trivial_comm::isend. Parks (data, size) under key {from, to, tag} without copying.
    void push( int from, int to, int tag, const void *data, int size )
    {
        auto res = messages_.emplace( key_type{ from, to, tag }, entry_type{ data, size } );

        if ( !res.second )
        {
            throw std::logic_error(
                "trivial_message_queue::push: message with this (from,to,tag) is already in flight"
            );
        }
    }

    /// Called by trivial_comm::irecv. Finds message {from, to, tag}, copies its bytes
    /// into dst via the Memory copy primitive and erases the entry. Returns false if no
    /// matching message is queued.
    bool recv( int from, int to, int tag, void *dst, int size )
    {
        key_type key_{ from, to, tag };

        auto it_ = messages_.find( key_ );
        if ( it_ != messages_.end() )
        {
            entry_type data_ = it_->second;
            assert( size == data_.size );
            mem_t::copy( data_.size, data_.data, dst );
            messages_.erase( it_ );

            return true;
        }

        return false;
    }

private:
    std::map<key_type, entry_type> messages_;
};

} // namespace detail
} // namespace communication
} // namespace scfd

#endif
