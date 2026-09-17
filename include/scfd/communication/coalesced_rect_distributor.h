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

#ifndef __SCFD_COALESCED_RECT_DISTRIBUTOR_H__
#define __SCFD_COALESCED_RECT_DISTRIBUTOR_H__

#include <cstddef>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include <scfd/arrays/tensorN_array_nd.h>
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
#    include <scfd/arrays/tensorN_array_nd_visible.h>
#endif
#include <scfd/for_each/for_each_func_macro.h>
#include "rect_partitioner.h"
#include "rect_distributor.h"

namespace scfd
{
namespace communication
{

template <class T, int Dim, class Memory, class ForEach, class Ord, class BigOrd, class Comm>
struct coalesced_rect_distributor
{
    /// NOTE these types for internal usage only
    typedef arrays::tensor1_array_nd<T, Dim, Memory, arrays::dyn_dim> array_type;

    typedef static_vec::vec<bool, Dim>               bool_vec_t;
    typedef static_vec::vec<Ord, Dim>                ord_vec_t;
    typedef static_vec::rect<Ord, Dim>               ord_rect_t;
    typedef static_vec::vec<BigOrd, Dim>             big_ord_vec_t;
    typedef static_vec::rect<BigOrd, Dim>            big_ord_rect_t;
    typedef rect_partitioner<Dim, Ord, BigOrd, Comm> rect_partitioner_t;
    typedef Comm                                     comm_type;
    typedef typename Comm::request_type              request_type;
    typedef typename Comm::status_type               status_type;

    coalesced_rect_distributor() = default;
    void init_for_tensors(
        Ord tensor_max_dim, const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags, Ord stencil_size,
        int stencil_max_order = 1
    )
    {
        ord_vec_t stencil_sizes = ord_vec_t::make_ones() * stencil_size;
        init_for_tensors(
            tensor_max_dim, partitioner, periodic_flags, stencil_sizes, stencil_sizes, stencil_max_order
        );
    }
    void init_for_tensors(
        Ord tensor_max_dim, const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags,
        ord_vec_t stencil_sizes, int stencil_max_order = 1
    )
    {
        init_for_tensors(
            tensor_max_dim, partitioner, periodic_flags, stencil_sizes, stencil_sizes, stencil_max_order
        );
    }
    void init_for_tensors(
        Ord tensor_max_dim, const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags,
        ord_vec_t stencil_sizes_lower, ord_vec_t stencil_sizes_upper, int stencil_max_order = 1
    )
    {
        tensor_max_dim_ = tensor_max_dim;
        comm_info_      = partitioner.comm_info;
        packets_in_by_rank_.resize( comm_info_.num_procs, nullptr );
        /// init packets_in
        for ( Ord sender_proc_id = 0; sender_proc_id < comm_info_.num_procs; ++sender_proc_id )
        {
            packet pack;
            init_packet(
                tensor_max_dim, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper,
                stencil_max_order, true, get_own_rank(), sender_proc_id, pack
            );
            if ( !pack.buckets.empty() )
            {
                pack.buf.init( pack.total_elems );
                packets_in_.emplace_back( std::move( pack ) );
            }
        }
        for ( auto &pkg : packets_in_ )
        {
            packets_in_by_rank_[pkg.proc_id] = &pkg;
        }
        irecv_requests_.resize( packets_in_.size() );
        /// init packets_out
        for ( Ord reciever_proc_id = 0; reciever_proc_id < comm_info_.num_procs; ++reciever_proc_id )
        {
            packet pack;
            init_packet(
                tensor_max_dim, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper,
                stencil_max_order, false, reciever_proc_id, get_own_rank(), pack
            );
            if ( !pack.buckets.empty() )
            {
                pack.buf.init( pack.total_elems );
                packets_out_.emplace_back( std::move( pack ) );
            }
        }
        isend_requests_.resize( packets_out_.size() );
    }
    void init(
        const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags, Ord stencil_size,
        int stencil_max_order = 1
    )
    {
        init_for_tensors( 1, partitioner, periodic_flags, stencil_size, stencil_max_order );
    }
    void init(
        const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags, ord_vec_t stencil_sizes,
        int stencil_max_order = 1
    )
    {
        init_for_tensors( 1, partitioner, periodic_flags, stencil_sizes, stencil_max_order );
    }
    void init(
        const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags, ord_vec_t stencil_sizes_lower,
        ord_vec_t stencil_sizes_upper, int stencil_max_order = 1
    )
    {
        init_for_tensors( 1, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper, stencil_max_order );
    }

    const Comm &comm_info() const
    {
        return comm_info_;
    }
    Ord get_own_rank() const
    {
        return comm_info_.myid;
    }

    template <class Array>
    void sync( const Array &array ) const
    {
        /// For now just create with default params
        ForEach for_each;

        //isend all
        for ( Ord pkg_i = 0; pkg_i < packets_out_.size(); ++pkg_i )
        {
            auto &pkg = packets_out_[pkg_i];
            pack_bucket_views( for_each, array, pkg );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            pkg.buf.sync_from_array();
#endif
            comm_info_.template isend<char>(
                transport_ptr( pkg ), static_cast<int>( bytes( pkg ) ), pkg.proc_id, 0, isend_requests_[pkg_i]
            );
        }
        //irecv all
        for ( Ord pkg_i = 0; pkg_i < packets_in_.size(); ++pkg_i )
        {
            auto &pkg = packets_in_[pkg_i];
            comm_info_.template irecv<char>(
                transport_ptr( pkg ), static_cast<int>( bytes( pkg ) ), pkg.proc_id, 0, irecv_requests_[pkg_i]
            );
        }
        //wait all irecv
        for ( Ord ireq = 0; ireq < static_cast<Ord>( packets_in_.size() ); ++ireq )
        {
            int         ireq_idx;
            status_type status;
            ireq_idx = comm_info_.waitany( packets_in_.size(), irecv_requests_.data(), &status );
            (void)ireq_idx;
            auto &pkg = *( packets_in_by_rank_[status.source()] );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            pkg.buf.sync_to_array();
#endif
            unpack_bucket_views( for_each, array, pkg );
        }
        //wait all isend
        comm_info_.waitall( isend_requests_.size(), isend_requests_.data() );
    }

private:
    struct bucket_desc
    {
        ord_rect_t  loc_rect;
        std::size_t elem_offset;
    };
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
    using buf_type = arrays::tensor1_array_nd_visible<T, 1, Memory, 1>;
#else
    using buf_type = arrays::tensor1_array_nd<T, 1, Memory, 1>;
#endif
    struct packet
    {
        Ord                      proc_id;
        std::vector<bucket_desc> buckets;
        std::size_t              total_elems = 0;
        buf_type                 buf;
    };

    Comm                comm_info_;
    Ord                 tensor_max_dim_ = 1;
    std::vector<packet> packets_in_, packets_out_;
    //TODO use indexes here instead of pointers
    std::vector<packet *>             packets_in_by_rank_;
    mutable std::vector<request_type> isend_requests_, irecv_requests_;

#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
    static T *device_ptr( const packet &pack )
    {
        return pack.buf.array().raw_ptr();
    }
    static char *transport_ptr( const packet &pack )
    {
        return (char *)pack.buf.raw_ptr();
    }
#else
    static T *device_ptr( const packet &pack )
    {
        return pack.buf.raw_ptr();
    }
    static char *transport_ptr( const packet &pack )
    {
        return (char *)pack.buf.raw_ptr();
    }
#endif
    static std::size_t bytes( const packet &pack )
    {
        return pack.total_elems * sizeof( T );
    }

    template <class Array>
    void pack_bucket_views( const ForEach &for_each, const Array &array, const packet &pkg ) const
    {
        if ( static_cast<Ord>( detail::get_array_tensor_dim( array ) ) > tensor_max_dim_ )
            throw std::logic_error(
                "coalesced_rect_distributor::pack_bucket_views: array tensor dim exceeds buffer tensor "
                "dim - incorrect distributor initialization"
            );
        for ( Ord bucket_i = 0; bucket_i < static_cast<Ord>( pkg.buckets.size() ); ++bucket_i )
        {
            auto      &bucket = pkg.buckets[bucket_i];
            array_type bucket_view;
            bucket_view.init_by_raw_data( device_ptr( pkg ) + bucket.elem_offset, bucket.loc_rect, tensor_max_dim_ );
            detail::copy_array1_nd_rect(
                for_each, static_cast<Ord>( detail::get_array_tensor_dim( array ) ),
                detail::array_as_tensor1_array( array ), bucket.loc_rect, bucket_view
            );
        }
    }
    template <class Array>
    void unpack_bucket_views( const ForEach &for_each, const Array &array, const packet &pkg ) const
    {
        if ( static_cast<Ord>( detail::get_array_tensor_dim( array ) ) > tensor_max_dim_ )
            throw std::logic_error(
                "coalesced_rect_distributor::unpack_bucket_views: array tensor dim exceeds buffer tensor "
                "dim - incorrect distributor initialization"
            );
        for ( Ord bucket_i = 0; bucket_i < static_cast<Ord>( pkg.buckets.size() ); ++bucket_i )
        {
            auto      &bucket = pkg.buckets[bucket_i];
            array_type bucket_view;
            bucket_view.init_by_raw_data( device_ptr( pkg ) + bucket.elem_offset, bucket.loc_rect, tensor_max_dim_ );
            detail::copy_array1_nd_rect(
                for_each, static_cast<Ord>( detail::get_array_tensor_dim( array ) ), bucket_view, bucket.loc_rect,
                detail::array_as_tensor1_array( array )
            );
        }
    }

    void init_packet(
        Ord tensor_max_dim, const rect_partitioner_t &partitioner, const bool_vec_t &periodic_flags,
        ord_vec_t stencil_sizes_lower, ord_vec_t stencil_sizes_upper, int stencil_max_order, bool is_in_pkg,
        Ord reciever_proc_id, Ord sender_proc_id, packet &pack
    )
    {
        big_ord_rect_t recv_rect = partitioner.proc_rects[reciever_proc_id],
                       send_rect = partitioner.proc_rects[sender_proc_id];
        big_ord_vec_t  my_i1     = partitioner.proc_rects[get_own_rank()].i1;
        if ( reciever_proc_id == get_own_rank() )
        {
            pack.proc_id = sender_proc_id;
        }
        else if ( sender_proc_id == get_own_rank() )
        {
            pack.proc_id = reciever_proc_id;
        }
        else
        {
            throw std::logic_error( "coalesced_rect_distributor::init_packet: niether sender nor reciever is my_id" );
        }
        for ( auto dir : big_ord_rect_t::make_symm_square_range() )
        {
            int stencil_order = 0;
            for ( int j = 0; j < Dim; ++j )
            {
                stencil_order += std::abs( dir[j] );
            }
            if ( ( stencil_order == 0 ) || ( stencil_order > stencil_max_order ) )
                continue;

            big_ord_vec_t padding_size;
            bool          empty_padding = false;
            for ( int j = 0; j < Dim; ++j )
            {
                if ( dir[j] < 0 )
                {
                    empty_padding   = empty_padding || ( stencil_sizes_lower[j] == 0 );
                    padding_size[j] = -stencil_sizes_lower[j];
                }
                else if ( dir[j] == 0 )
                {
                    padding_size[j] = 0;
                }
                else
                {
                    empty_padding   = empty_padding || ( stencil_sizes_upper[j] == 0 );
                    padding_size[j] = stencil_sizes_upper[j];
                }
            }
            if ( empty_padding )
                continue;
            big_ord_rect_t stencil_rect_base = recv_rect.padding_rect( padding_size );

            auto periodic_flags_rect = big_ord_rect_t::make_square_range();
            /// Exclude non-periodic directions from tests
            for ( int j = 0; j < Dim; ++j )
            {
                if ( ( !periodic_flags[j] ) || ( dir[j] == 0 ) )
                    periodic_flags_rect.i2[j] = 1;
            }
            for ( auto periodic_flags_test : periodic_flags_rect )
            {
                big_ord_vec_t periodic_shift =
                    pointwise_prod( periodic_flags_test, pointwise_prod( -dir, partitioner.dom_size ) );
                big_ord_rect_t stencil_rect = stencil_rect_base.shifted( periodic_shift );

                big_ord_rect_t common_rect_from_recv = stencil_rect.intersect( send_rect ),
                               common_rect_from_send = common_rect_from_recv;
                if ( !common_rect_from_recv.is_empty() )
                {
                    common_rect_from_recv = common_rect_from_recv.shifted( -periodic_shift );
                }
                /// TODO how to check or proove that only one of these cases is met at once
                if ( common_rect_from_recv.is_empty() != common_rect_from_send.is_empty() )
                    throw std::logic_error(
                        "coalesced_rect_distributor::init_packet: common_rect send recv is_empty differs"
                    );
                if ( common_rect_from_recv.is_empty() )
                    continue;
                big_ord_rect_t common_rect_loc = ( is_in_pkg ? common_rect_from_recv : common_rect_from_send );
                common_rect_loc.i1 -= my_i1;
                common_rect_loc.i2 -= my_i1;
                ord_rect_t  loc_rect( common_rect_loc.i1, common_rect_loc.i2 );
                std::size_t bucket_elems =
                    static_cast<std::size_t>( loc_rect.calc_area() ) * static_cast<std::size_t>( tensor_max_dim );
                pack.buckets.push_back( bucket_desc{ loc_rect, pack.total_elems } );
                pack.total_elems += bucket_elems;
            }
        }
    }
};

} // namespace communication
} // namespace scfd

#endif
