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

#ifndef __SCFD_PARTED_MAP_H__
#define __SCFD_PARTED_MAP_H__

#include <stdio.h>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <string>
#include <map>

namespace scfd
{
namespace communication
{

//supposed to satisfy MAP concept
//Partitioner must satisfy Partitioner concept
template<class Partitioner>
struct parted_map
{
    using loc_ordinal_type = int;
    using glob_ordinal_type = typename Partitioner::glob_ordinal_type;

    Partitioner                                         part;
    bool                                                r_stencil_only;
    std::map<glob_ordinal_type,loc_ordinal_type>        stencil_glob_2_loc;
    std::vector<glob_ordinal_type>                      l_stencil_loc_2_glob;
    std::vector<glob_ordinal_type>                      r_stencil_loc_2_glob;

    //creates uninitialized map
    parted_map() {}
    //dummy constructor
    parted_map(const Partitioner &_part,bool _r_stencil_only = false) : part(_part), r_stencil_only(_r_stencil_only)
    {                
    }

    //'read' before construction part:
    int                     get_own_rank()const { return part.get_own_rank(); }
    bool                    check_glob_owned(int i)const { return part.check_glob_owned(i); }
    glob_ordinal_type       get_total_size()const { return part.get_total_size(); }
    loc_ordinal_type        get_size()const { return part.get_size(); }
    glob_ordinal_type       own_glob_ind(loc_ordinal_type i)const { return part.own_glob_ind(i); }

    //'construction' part:
    //t_simple_map just ignores this information
    //i is global index here
    void    add_stencil_element(glob_ordinal_type i)
    {
        if (check_glob_owned(i)) return;
        part.add_stencil_element(i);
        //before complete() stencil_glob_2_loc simply store all stencil elements global indices
        stencil_glob_2_loc[i] = 0;
    }
    void    complete()
    {
        part.complete();
        glob_ordinal_type median_elem = 0;
        if (get_size() > 0) 
        {
            if (!r_stencil_only) median_elem = own_glob_ind(0); else median_elem = 0;
        }
        //typedef std::pair<const int,int> pair_int_int;
        for (const auto &e : stencil_glob_2_loc) 
        {
            if (e.first < median_elem) 
            {
                //to the left
                l_stencil_loc_2_glob.push_back(e.first);
            } 
            else 
            {
                //to the right
                r_stencil_loc_2_glob.push_back(e.first);
            } 
        }
        std::sort(l_stencil_loc_2_glob.begin(),l_stencil_loc_2_glob.end());
        std::sort(r_stencil_loc_2_glob.begin(),r_stencil_loc_2_glob.end());
        for (loc_ordinal_type i = 0;i < l_stencil_loc_2_glob.size();++i) 
        {
            glob_ordinal_type     glob_ind = l_stencil_loc_2_glob[i];
            loc_ordinal_type      loc_ind = -l_stencil_loc_2_glob.size() + i;
            stencil_glob_2_loc[glob_ind] = loc_ind;
        }
        for (loc_ordinal_type i = 0;i < r_stencil_loc_2_glob.size();++i) 
        {
            glob_ordinal_type     glob_ind = r_stencil_loc_2_glob[i];
            loc_ordinal_type      loc_ind = get_size() + i;
            stencil_glob_2_loc[glob_ind] = loc_ind;
        }
    }

    //'read' after construction part:
    int                 get_rank(int i)const { return part.get_rank(i); }
    bool                check_glob_owned(int i, int rank)const { return part.check_glob_owned(i, rank); }
    glob_ordinal_type   loc2glob(loc_ordinal_type i_loc)const
    {
        if (i_loc < 0) 
        {
            return l_stencil_loc_2_glob[i_loc + l_stencil_loc_2_glob.size()];
        } 
        else if (i_loc >= get_size()) 
        {
            return r_stencil_loc_2_glob[i_loc - get_size()];
        } 
        else 
        {
            return part.own_glob_ind(i_loc);
        }
    }
    loc_ordinal_type    glob2loc(glob_ordinal_type i_glob)const
    {
        if (check_glob_owned(i_glob))
            return part.own_glob_ind_2_ind(i_glob);
        else
            return stencil_glob_2_loc.at(i_glob);
            //return stencil_glob_2_loc[i_glob];
    }
    loc_ordinal_type    own_loc_ind(loc_ordinal_type i)const
    {
        return i;
    }
    loc_ordinal_type    min_loc_ind()const
    {
        return -l_stencil_loc_2_glob.size();
    }
    loc_ordinal_type    max_loc_ind()const
    {
        return r_stencil_loc_2_glob.size()-1 +  get_size();
    }
    loc_ordinal_type    min_own_loc_ind()const
    {
        return 0;
    }
    loc_ordinal_type    max_own_loc_ind()const
    {
        return get_size()-1;
    }
    bool    check_glob_has_loc_ind(glob_ordinal_type i_glob)const
    {
        if (check_glob_owned(i_glob)) return true;
        return stencil_glob_2_loc.find(i_glob) != stencil_glob_2_loc.end();
    }
    bool    check_loc_has_loc_ind(loc_ordinal_type i_loc)const
    {
        return (i_loc >= min_loc_ind())&&(i_loc <= max_loc_ind());
    }
    bool  is_loc_glob_ind_order_preserv()const
    {
        return true;
    }
};

}  /// namespace communication
}  /// namespace scfd

#endif
