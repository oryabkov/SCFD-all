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

#ifndef __SCFD_NODES_PARTITIONER_H__
#define __SCFD_NODES_PARTITIONER_H__

#include <cassert>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace scfd
{
namespace communication
{

//WARNING i think we will reqiure MPI here!

//supposed to satisfy PARTITIONER concept

template<class Ord>
struct faces_partitioner
{
    using glob_ordinal_type = Ord;

    Ord                             total_size;
    int                             my_rank;
    bool                            is_complete;
    //first Ord is global index of face, second int is rank
    std::unordered_map<Ord,int>     ranks;
    //supposed to be sorted
    std::vector<Ord>                own_glob_indices;
    std::unordered_map<Ord,Ord>     own_glob_indices_2_ind;

    faces_partitioner() {}
    //map_elem could be at 'before complete' stage
    //mesh must have all nodes incident(to owned by map_elen) and 2nd order incident (to owned by map_elem) elements nodes-elem graph
    template<class Communicator, class HostMesh, class Map>
    faces_partitioner(const Communicator &comm, const HostMesh &mesh, const Map &map_elem) : my_rank(comm.my_rank()), is_complete(false)
    {
        //total_size = mesh.nodes.size();
        for (Ord i = 0;i < map_elem.get_size();++i) 
        {
            Ord elem_glob_i = map_elem.own_glob_ind(i);

            Ord faces[mesh.get_elem_faces_num(elem_glob_i)];
            mesh.get_elem_faces(elem_glob_i, faces);

            for (Ord face_i = 0;face_i < mesh.get_elem_faces_num(elem_glob_i);++face_i) 
            {
                //Ord node_i = mesh.cv_2_node_ids[elem_glob_i].ids[vert_i];
                if (ranks.find(faces[face_i]) != ranks.end()) continue;

                Ord face_elems[2];
                mesh.get_face_elems(faces[face_i],face_elems);

                //std::pair<int,int> ref_range = mesh.node_2_cv_ids_ref[node_i];
                Ord min_elem_id = face_elems[0];
                for (Ord j = 0;j < mesh.get_face_elems_num(faces[face_i]);++j) 
                {
                    if (face_elems[j] < min_elem_id) min_elem_id = face_elems[j];
                }
                if (map_elem.check_glob_owned(min_elem_id)) 
                {
                    ranks[faces[face_i]] = my_rank;
                    own_glob_indices.push_back(faces[face_i]);
                }
            }
        }
        std::sort(own_glob_indices.begin(), own_glob_indices.end());
        for (Ord i = 0;i < own_glob_indices.size();++i) 
        {
            own_glob_indices_2_ind[ own_glob_indices[i] ] = i;
        }

        Ord own_size = own_glob_indices.size();

        total_size = comm.template reduce_sum<Ord>(own_size);

        //my_rank = part.my_rank;
        //is_complete = part.is_complete;
    }

    //'read' before construction part:
    int     get_own_rank()const { return my_rank; }
    //i_glob here could be any global index both before and after construction part
    bool    check_glob_owned(Ord i_glob)const 
    {
        auto  it = ranks.find(i_glob);
        if (it == ranks.end()) return false;
        return it->second == get_own_rank();
    }
    Ord     get_total_size()const { return total_size; }
    //get_size is not part of PARTITIONER concept
    Ord     get_size()const { return own_glob_indices.size(); }
    Ord     own_glob_ind(Ord i)const
    {
        assert((i >= 0)&&(i < get_size()));
        return own_glob_indices[i];
    }
    Ord     own_glob_ind_2_ind(Ord i_glob)const
    {
        auto  it = own_glob_indices_2_ind.find(i_glob);
        assert(it != own_glob_indices_2_ind.end());
        return it->second;
    }

    //'construction' part:
    void    add_stencil_element(Ord i_glob)
    {
        if (check_glob_owned(i_glob)) return;
        ranks[i_glob] = -1;
    }
    void    complete()
    {
        is_complete = true;
        //TODO i think the only way is to use mpi communications here
    }

    //'read' after construction part:
    //i_glob is ether index owner by calling process, ether index from stencil, otherwise behavoiur is  undefined
    //returns rank of process that owns i_glob index
    int     get_rank(Ord i_glob)const
    {
        assert(is_complete);
        auto it = ranks.find(i_glob);
        assert(it != ranks.end());
        if (it->second == -1) throw std::logic_error("faces_partitioner:: not realized yet!!");
        return it->second;
    }
    //for (rank == get_own_rank()) result and behavoiur coincides with check_glob_owned(i_glob)
    //for rank != get_own_rank():
    //i_glob is ether index owner by calling process, ether index from stencil, otherwise behavoiur is  undefined
    //returns, if index i_glob is owned by rank processor
    bool    check_glob_owned(Ord i_glob, int rank)const 
    { 
        assert(is_complete);
        if (rank == get_own_rank()) {
            return check_glob_owned(i_glob);
        } else {
            return get_rank(i_glob) == rank; 
        }
    }
};

}  /// namespace communication
}  /// namespace scfd

#endif
