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

#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensorN_array_nd.h>
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
#include <scfd/arrays/tensorN_array_nd_visible.h>
#endif
#include <scfd/for_each/for_each_func_macro.h>
#include "mpi_comm_info.h"
#include "rect_partitioner.h"

//TODO:
//1.SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI must be replaced with template boolean parameter (device_aware_comm - something like this)
//2.explicit calls to MPI_ should be replaced with calls to methods of mpi_comm_info (isend,irecv,wait should be added into comm before doing this)
//3.mpi_rect_distributor implementation should be moved into rect_distributor class with Comm template parameter, mpi_rect_distributor may be left as alias with mpi_comm_info
//4.to do 3 rect_partitioner must be equipped with CommInfo template parameter
//5.need to avoid extra data send when actual synced array dimension is lesser then max tensor dim during initialization
//6.ideally adaptive enlargment of buffers tensor size should be added
//7.tensor2/3/4 support should be added

namespace scfd
{
namespace communication
{

namespace detail
{
namespace kernel
{

template<int Dim,class Ord,class Array>
struct copy_array1_nd_func 
{
    SCFD_FOR_EACH_FUNC_PARAMS(
        copy_array1_nd_func,
        Ord, tensor_dim, Array, input, Array, output
    )
    __DEVICE_TAG__ void operator()(const static_vec::vec<Ord,Dim> &idx)
    {
        for (Ord j = 0;j < tensor_dim;++j)
            output(idx,j) = input(idx,j);
    }
};

} // namespace kernel

template<class ForEach,int Dim,class Ord, class Array>    
void copy_array1_nd_rect(
    const ForEach &for_each, 
    Ord tensor_dim, const Array &input, const static_vec::rect<Ord,Dim> &rect, 
    const Array &output
)
{   
    for_each(
        kernel::copy_array1_nd_func<Dim,Ord,Array>(tensor_dim, input, output),
        rect
    );
}

template<class T,int Dim,class Memory>
int get_array_tensor_dim(const arrays::array_nd<T,Dim,Memory> &array)
{
    return 1;
}

template<class T,int Dim,class Memory,int TensorDim>
int get_array_tensor_dim(const arrays::tensor1_array_nd<T,Dim,Memory,TensorDim> &array)
{
    //TODO add get_tensor_dim<0> into array?
    return array.template get_dim<Dim>();
}

template<class T,int Dim,class Memory>
arrays::tensor1_array_nd<T,Dim,Memory,arrays::dyn_dim> array_as_tensor1_array(const arrays::array_nd<T,Dim,Memory> &array)
{
    //TODO add corresponding constructor into arrays
    arrays::tensor1_array_nd<T,Dim,Memory,arrays::dyn_dim> res;
    res.init_by_raw_data(array.raw_ptr(),array.rect_nd(),get_array_tensor_dim(array));
    return res;
}

//TODO seems we dont need separate implementations as implementation is the same. use Array template mb?

template<class T,int Dim,class Memory,int TensorDim>
arrays::tensor1_array_nd<T,Dim,Memory,arrays::dyn_dim> array_as_tensor1_array(const arrays::tensor1_array_nd<T,Dim,Memory,TensorDim> &array)
{
    arrays::tensor1_array_nd<T,Dim,Memory,arrays::dyn_dim> res;
    res.init_by_raw_data(array.raw_ptr(),array.rect_nd(),get_array_tensor_dim(array));
    return res;
}

} // namespace detail

template<class T,int Dim,class Memory,class ForEach,class Ord,class BigOrd>
struct mpi_rect_distributor
{
    /// NOTE these types for internal usage only
    typedef     arrays::tensor1_array_nd<T,Dim,Memory,arrays::dyn_dim>          array_type;
    #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
    typedef     arrays::tensor1_array_nd_visible<T,Dim,Memory,arrays::dyn_dim>  vis_array_type;
    #endif

    ///TODO rename with _type suffix
    typedef     static_vec::vec<bool,Dim>               bool_vec_t;
    typedef     static_vec::vec<Ord,Dim>                ord_vec_t;
    typedef     static_vec::rect<Ord,Dim>               ord_rect_t;
    typedef     static_vec::vec<BigOrd,Dim>             big_ord_vec_t;
    typedef     static_vec::rect<BigOrd,Dim>            big_ord_rect_t;
    typedef     rect_partitioner<Dim,Ord,BigOrd>        rect_partitioner_t;

    mpi_rect_distributor() = default;
    void init_for_tensors(
        Ord tensor_max_dim,
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, Ord stencil_size, int stencil_max_order = 1
    )
    {
        ord_vec_t stencil_sizes = ord_vec_t::make_ones()*stencil_size;
        init_for_tensors(tensor_max_dim, partitioner, periodic_flags, stencil_sizes, stencil_sizes, stencil_max_order);
    }
    void init_for_tensors(
        Ord tensor_max_dim,
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, 
        ord_vec_t stencil_sizes,
        int stencil_max_order = 1
    )
    {
        init_for_tensors(tensor_max_dim, partitioner, periodic_flags, stencil_sizes, stencil_sizes, stencil_max_order);
    }
    void init_for_tensors(
        Ord tensor_max_dim,
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, 
        ord_vec_t stencil_sizes_lower, 
        ord_vec_t stencil_sizes_upper,
        int stencil_max_order = 1
    )
    {
        comm_info_ = partitioner.comm_info;
        packets_in_by_rank_.resize(comm_info_.num_procs, nullptr);
        /// init packets_in
        for (Ord sender_proc_id = 0;sender_proc_id < comm_info_.num_procs;++sender_proc_id)
        {
            packet pack;
            init_packet(
                tensor_max_dim, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper,
                stencil_max_order, true, get_own_rank(), sender_proc_id, pack
            );
            if (!pack.buckets.empty())
            {
                packets_in_.emplace_back(std::move(pack));
            }
        }
        for (auto &pkg : packets_in_)
        {
            packets_in_by_rank_[pkg.proc_id] = &pkg;
        }
        irecv_requests_.resize(calc_buckets_num(packets_in_));
        /// init packets_out
        for (Ord reciever_proc_id = 0;reciever_proc_id < comm_info_.num_procs;++reciever_proc_id)
        {
            packet pack;
            init_packet(
                tensor_max_dim, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper,
                stencil_max_order, false, reciever_proc_id, get_own_rank(), pack
            );
            if (!pack.buckets.empty())
            {
                packets_out_.emplace_back(std::move(pack));
            }
        }
        isend_requests_.resize(calc_buckets_num(packets_out_));
    }
    void init(
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, Ord stencil_size, int stencil_max_order = 1
    )
    {
        init_for_tensors(1, partitioner, periodic_flags, stencil_size, stencil_max_order);
    }
    void init(
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, 
        ord_vec_t stencil_sizes,
        int stencil_max_order = 1
    )
    {
        init_for_tensors(1, partitioner, periodic_flags, stencil_sizes, stencil_max_order);
    }
    void init(
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, 
        ord_vec_t stencil_sizes_lower, 
        ord_vec_t stencil_sizes_upper,
        int stencil_max_order = 1
    )
    {
        init_for_tensors(1, partitioner, periodic_flags, stencil_sizes_lower, stencil_sizes_upper, stencil_max_order);
    }
    
    const mpi_comm_info  &comm_info()const
    {
        return comm_info_;
    }
    Ord     get_own_rank()const { return comm_info_.myid; }

    template<class Array>
    void    sync(const Array &array)const
    {
        /// For now just create with default params
        ForEach for_each;

        Ord isend_requests_count = 0, 
            irecv_requests_count = 0;

        //isend all
        //std::cout << "isend all " << std::endl;
        for (Ord pkg_i = 0;pkg_i < packets_out_.size();++pkg_i)
        {
            auto &pkg = packets_out_[pkg_i];
            for (Ord bucket_i = 0;bucket_i < pkg.buckets.size();++bucket_i)
            {
                auto &bucket = pkg.buckets[bucket_i];
                bucket.sync_from_array(for_each, array);
                auto mpi_res = 
                    MPI_Isend(
                        bucket.buf(), bucket.buf_size(), MPI_BYTE, 
                        pkg.proc_id, bucket_i, comm_info_.comm, 
                        &(isend_requests_[isend_requests_count])
                    );
                if (mpi_res != MPI_SUCCESS)
                    throw std::runtime_error("mpi_rect_distributor::sync:MPI_Isend failed");
                isend_requests_count++;
            }
        }
        //irecv all
        //std::cout << "irecv all " << std::endl;
        for (Ord pkg_i = 0;pkg_i < packets_in_.size();++pkg_i)
        {
            auto &pkg = packets_in_[pkg_i];
            for (Ord bucket_i = 0;bucket_i < pkg.buckets.size();++bucket_i)
            {
                auto &bucket = pkg.buckets[bucket_i];
                //TODO check if bucket_i is actually message tag
                auto mpi_res = 
                    MPI_Irecv(
                        bucket.buf(), bucket.buf_size(), MPI_BYTE, 
                        pkg.proc_id, bucket_i, comm_info_.comm, 
                        &(irecv_requests_[irecv_requests_count])
                    );
                if (mpi_res != MPI_SUCCESS)
                    throw std::runtime_error("mpi_rect_distributor::sync:MPI_Irecv failed");
                irecv_requests_count++;
            }
        }
        //wait all irecv
        //std::cout << "wait all irecv" << std::endl;
        for (Ord ireq = 0;ireq < irecv_requests_count;++ireq) 
        {
            //std::cout << "ireq = " << ireq << std::endl;
            int             ireq_idx;
            MPI_Status      status;
            auto mpi_res = MPI_Waitany( irecv_requests_count, irecv_requests_.data(), &ireq_idx, &status);
            if (mpi_res != MPI_SUCCESS) 
                throw std::runtime_error("mpi_rect_distributor::sync:MPI_Waitany failed");
            //std::cout << "after wait any ireq = " << ireq << " status.MPI_SOURCE = " << status.MPI_SOURCE << " status.MPI_TAG = " << status.MPI_TAG << std::endl;
            //std::cout << "packets_in_by_rank_[status.MPI_SOURCE] = " << packets_in_by_rank_[status.MPI_SOURCE] << std::endl;
            auto &pkg = *(packets_in_by_rank_[status.MPI_SOURCE]);
            //std::cout << "pkg.buckets.size() = " << pkg.buckets.size() << std::endl;
            auto &bucket = pkg.buckets[status.MPI_TAG];
            //TODO check for an error MPI_ERROR?
            //std::cout << "before sync_to_array" << std::endl;
            bucket.sync_to_array(for_each, array);
            //std::cout << "after sync_to_array" << std::endl;
        }
        //wait all isend
        //std::cout << "wait all isend" << std::endl;
        if (MPI_Waitall( isend_requests_count, isend_requests_.data(), MPI_STATUS_IGNORE) != MPI_SUCCESS) 
            throw std::runtime_error("mpi_rect_distributor::sync:MPI_Waitall failed");
    }
private:
    /// packet_bucket and packet works both for send and recv
    struct packet_bucket
    {
        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        using buf_array_type = vis_array_type;
        #else
        using buf_array_type = array_type;
        #endif
        ord_rect_t                       loc_rect;
        /// TODO Would be nice to have unique_array's
        std::unique_ptr<buf_array_type>  data_buf;

        packet_bucket(const ord_rect_t &loc_rect_p, Ord tensor_max_dim) : 
          loc_rect(loc_rect_p),
          data_buf(std::make_unique<buf_array_type>())
        {
            data_buf->init(loc_rect.i2-loc_rect.i1,loc_rect.i1,tensor_max_dim);
        }

        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        array_type  buf_array_device()const
        {
            return data_buf->array();
        }
        #else
        array_type  buf_array_device()const
        {
            return *data_buf;
        }
        #endif
        char        *buf()const
        {
            return (char*)data_buf->raw_ptr();
        }
        std::size_t buf_size()const
        {
            return data_buf->total_size()*sizeof(T);
        }
        template<class Array>
        void        sync_from_array(const ForEach &for_each, const Array &array)const
        {
            if (detail::get_array_tensor_dim(array) > detail::get_array_tensor_dim(buf_array_device()))
                throw std::logic_error("mpi_rect_distributor::packet_bucket::sync_from_array: array tensor dir exceeds buffer tensor dim - incorrect distributor initialization");

            detail::copy_array1_nd_rect(
                for_each, detail::get_array_tensor_dim(array), detail::array_as_tensor1_array(array), loc_rect, buf_array_device()
            );
            #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            data_buf->sync_from_array();
            #endif
        }
        template<class Array>
        void        sync_to_array(const ForEach &for_each, const Array &array)const
        {
            if (detail::get_array_tensor_dim(array) > detail::get_array_tensor_dim(buf_array_device()))
                throw std::logic_error("mpi_rect_distributor::packet_bucket::sync_to_array: array tensor dir exceeds buffer tensor dim - incorrect distributor initialization");

            #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            data_buf->sync_to_array();
            #endif
            //std::cout << "sync_to_array" << std::endl;
            //auto i1 = loc_rect.i1,i2 = loc_rect.i2;
            //ord_vec_t i1,i2;
            auto array_sz = array.size_nd(), array_i1 = array.indexes0_nd();
            //std::cout << "array: sz = " << array_sz[0] << "," << array_sz[1] << "," << array_sz[2] << std::endl;
            //std::cout << "array: i1 = " << array_i1[0] << "," << array_i1[1] << "," << array_i1[2] << std::endl;
            //std::cout << "loc_rect: i1 = " << i1[0] << "," << i1[1] << "," << i1[2] << std::endl;
            //std::cout << "loc_rect: i2 = " << i2[0] << "," << i2[1] << "," << i2[2] << std::endl;
            detail::copy_array1_nd_rect(
                for_each, detail::get_array_tensor_dim(array), buf_array_device(), loc_rect, detail::array_as_tensor1_array(array)
            );
        }
    };
    struct packet
    {
        Ord                         proc_id;
        std::vector<packet_bucket>  buckets;
    };

    mpi_comm_info                       comm_info_;
    std::vector<packet>                 packets_in_, packets_out_;
    //TODO use indexes here instead of pointers
    std::vector<packet*>                packets_in_by_rank_;
    mutable std::vector<MPI_Request>    isend_requests_, irecv_requests_;

    void init_packet(
        Ord tensor_max_dim,
        const rect_partitioner_t &partitioner,
        const bool_vec_t &periodic_flags, 
        ord_vec_t stencil_sizes_lower, ord_vec_t stencil_sizes_upper,
        int stencil_max_order,
        bool is_in_pkg, Ord reciever_proc_id, Ord sender_proc_id, packet &pack
    )
    {
        big_ord_rect_t  recv_rect = partitioner.proc_rects[reciever_proc_id],
                        send_rect = partitioner.proc_rects[sender_proc_id];
        big_ord_vec_t   my_i1 = partitioner.proc_rects[get_own_rank()].i1;
        if (reciever_proc_id == get_own_rank())
        {
            pack.proc_id = sender_proc_id;
        } else if (sender_proc_id == get_own_rank())
        {
            pack.proc_id = reciever_proc_id;
        } 
        else 
        {
            throw std::logic_error("mpi_rect_distributor::init_packet: niether sender nor reciever is my_id");
        }
        for (auto dir : big_ord_rect_t::make_symm_square_range())
        {
            int stencil_order = 0;
            for (int j = 0;j < Dim;++j)
            {
                stencil_order += std::abs(dir[j]);
            }
            if (stencil_order > stencil_max_order) continue;

            big_ord_vec_t padding_size;
            for (int j = 0;j < Dim;++j)
            {
                if (dir[j] < 0)
                    padding_size[j] = -stencil_sizes_lower[j];
                else if (dir[j] == 0)
                    padding_size[j] = 0;
                else
                    padding_size[j] = stencil_sizes_upper[j];
            }
            big_ord_rect_t  stencil_rect_base = recv_rect.padding_rect(padding_size);

            auto periodic_flags_rect = big_ord_rect_t::make_square_range();
            /// Exclude non-periodic directions from tests
            for (int j = 0;j < Dim;++j)
            {
                if (!periodic_flags[j]) periodic_flags_rect.i2[j] = 1;
            }
            for (auto periodic_flags_test : periodic_flags_rect)
            {
                big_ord_vec_t periodic_shift = pointwise_prod(periodic_flags_test, pointwise_prod(-dir,partitioner.dom_size));
                big_ord_rect_t  stencil_rect = stencil_rect_base.shifted(periodic_shift);
                
                /*if (sign == -1)
                {
                    stencil_rect.i2[j] = stencil_rect.i1[j];
                    stencil_rect.i1[j] = stencil_rect.i1[j]-stencil_sizes[j];
                }
                else
                {
                    stencil_rect.i1[j] = stencil_rect.i2[j];
                    stencil_rect.i2[j] = stencil_rect.i2[j]+stencil_sizes[j];
                }*/
                big_ord_rect_t  common_rect_from_recv = stencil_rect.intersect(send_rect),
                                common_rect_from_send = common_rect_from_recv;
                /// Try periodic case if not
                //if (common_rect_from_recv.is_empty() && periodic_flags[j])
               // {
                    
                    //stencil_rect.i1[j] += periodic_shift;
                    //stencil_rect.i2[j] += periodic_shift;
                    //common_rect_from_send = stencil_rect.intersect(send_rect);
                    //common_rect_from_recv = common_rect_from_send;
                    
                //}
                if (!common_rect_from_recv.is_empty())
                {
                    common_rect_from_recv = common_rect_from_recv.shifted(-periodic_shift);

                    //common_rect_from_recv.i1[j] -= periodic_shift;
                    //common_rect_from_recv.i2[j] -= periodic_shift;
                }
                /// TODO how to check or proove that only one of these cases is met at once
                if (common_rect_from_recv.is_empty() != common_rect_from_send.is_empty()) 
                    throw std::logic_error("mpi_rect_distributor::init_packet: common_rect send recv is_empty differs");
                if (common_rect_from_recv.is_empty()) continue;
                big_ord_rect_t  common_rect_loc = (is_in_pkg?common_rect_from_recv:common_rect_from_send);
                common_rect_loc.i1 -= my_i1;
                common_rect_loc.i2 -= my_i1;
                pack.buckets.emplace_back(ord_rect_t(common_rect_loc.i1,common_rect_loc.i2),tensor_max_dim);
            }
        }
    }

    Ord calc_buckets_num(const std::vector<packet> &packets)const
    {
        Ord res(0);
        for (const auto &pkg : packets)
        {
            res += pkg.buckets.size();
        }
        return res;
    }
};

} // namespace communication
} // namespace scfd

#endif
