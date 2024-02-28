#ifndef __SCFD_RECT_PARTITIONER_H__
#define __SCFD_RECT_PARTITIONER_H__

#include <vector>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include "mpi_communicator_info.h"

namespace scfd
{
namespace communication
{

template<class Ord,class BigOrd,int Dim>
struct rect_partitioner
{
    typedef     static_vec::vec<BigOrd,Dim>          big_ord_vec_t;
    typedef     static_vec::rect<BigOrd,Dim>         big_ord_rect_t;

    mpi_communicator_info<Ord>      comm_info;
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
    rect_partitioner(const mpi_communicator_info<Ord> &comm_info_p, const big_ord_vec_t &dom_size_p) : 
      comm_info(comm_info_p), dom_size(dom_size_p)
    {
        //TODO add some trivial decomposition initialization
    }
    
    Ord     get_own_rank()const { return comm_info.myid; }
};

} // namespace communication
} // namespace scfd

#endif
