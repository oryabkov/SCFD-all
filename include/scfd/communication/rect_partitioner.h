#ifndef __SCFD_RECT_PARTITIONER_H__
#define __SCFD_RECT_PARTITIONER_H__

#include <vector>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include "mpi_comm_info.h"

namespace scfd
{
namespace communication
{

template<int Dim,class Ord,class BigOrd>
struct rect_partitioner
{
    typedef     static_vec::vec<Ord,Dim>             ord_vec_t;
    typedef     static_vec::rect<Ord,Dim>            ord_rect_t;
    typedef     static_vec::vec<BigOrd,Dim>          big_ord_vec_t;
    typedef     static_vec::rect<BigOrd,Dim>         big_ord_rect_t;

    mpi_comm_info                   comm_info;
    //proc_rects is of size comm_size
    //proc_rects[i] contains indexes rect owned by i-th process
    big_ord_vec_t                   dom_size;
    std::vector<big_ord_rect_t>     proc_rects;
    //big_ord_rect_t                  own_rect_glob, own_rect_loc;
    //big_ord_rect_t                  own_rect_no_stencil_glob, own_rect_no_stencil_loc;
    //global coordinates of point with (0,0,0) local coordinates
    //big_ord_vec_t                   loc0_glob;


    /*big_ord_vec_t               loc2glob_vec(const big_ord_vec_t &v_loc)const
    {
        return loc0_glob + v_loc;
    }
    big_ord_vec_t               glob2loc_vec(const big_ord_vec_t &v_glob)const
    {
        return v_glob - loc0_glob;
    }*/
    /*Ord                     get_rank_vec(const big_ord_vec_t &v_glob)const 
    { 
        for (Ord rank = 0;rank < comm_size;++rank) {
            if (check_glob_owned_vec(v_glob, rank)) return rank;
        }
        throw std::logic_error("rect_map::get_rank_vec: vector does not belong to any proc");
        return -1;
    }*/
    /*bool                    check_glob_owned_vec(const big_ord_vec_t &v_glob, Ord rank)const
    {
        return proc_rects_glob[rank].is_own(v_glob);
    }
    bool                    check_glob_owned_vec(const big_ord_vec_t &v_glob)const
    {
        return check_glob_owned_vec(v_glob, my_rank);
    }*/

    rect_partitioner() = default;
    rect_partitioner(const mpi_comm_info &comm_info_p, const big_ord_vec_t &dom_size_p) : 
      comm_info(comm_info_p), dom_size(dom_size_p), proc_rects(comm_info.num_procs)
    {
        std::vector<BigOrd>  lin_sizes(comm_info.num_procs,dom_size[0]/comm_info.num_procs);
        BigOrd rem_size = dom_size[0]%comm_info.num_procs;
        for (BigOrd i = 0;i < rem_size;++i)
        {
            lin_sizes[i]++;
        }
        BigOrd curr_lin_idx = 0;
        for (Ord ip = 0;ip < comm_info.num_procs;++ip)
        {
            proc_rects[ip] = big_ord_rect_t(dom_size);
            proc_rects[ip].i1[0] = curr_lin_idx;
            curr_lin_idx += lin_sizes[ip];
            proc_rects[ip].i2[0] = curr_lin_idx;
        }
    }
    
    Ord             get_own_rank()const { return comm_info.myid; }
    big_ord_rect_t  get_own_rect()const
    {
        return proc_rects[comm_info.myid];
    }
    ord_rect_t      get_own_loc_rect()const
    {
        auto my_glob_rect = get_own_rect();
        return my_glob_rect.shifted(-my_glob_rect.i1);
    }
    big_ord_rect_t  get_dom_rect()const
    {
        return big_ord_rect_t(big_ord_vec_t::make_zero(),dom_size);
    }
};

} // namespace communication
} // namespace scfd

#endif
