// Copyright © 2016,2017 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_SERIAL_MAP_H__
#define __SCFD_SERIAL_MAP_H__

//supposed to satisfy MAP concept

namespace scfd
{
namespace communication
{

struct serial_map
{
    int         total_size;

    //creates uninitialized map
    serial_map() {}
    //dummy constructor
    serial_map(int N)
    {
        total_size = N;
    }

    //'read' before construction part:
    int     get_own_rank()const { return 0; }
    bool    check_glob_owned(int i)const { return check_glob_owned(i, get_own_rank()); }
    int     get_total_size()const { return total_size; }
    int     get_size()const { return total_size; }
    //in this map 'i' in own_glob_ind and own_loc_ind simply coincides with local index (not true for general MAP)
    int     own_glob_ind(int i)const { return i; }

    //'construction' part:
    //t_simple_map just ignores this information
    void    add_stencil_element(int i) { }
    void    complete() { }

    //'read' after construction part:
    int     get_rank(int i)const { return 0; }
    bool    check_glob_owned(int i, int rank)const { return ((i >= 0)&&(i < total_size)); }
    int     loc2glob(int i_loc)const { return i_loc; }
    int     glob2loc(int i_glob)const { return i_glob; }
    int     own_loc_ind(int i)const { return i; }
    int     min_loc_ind()const { return 0; }
    int     max_loc_ind()const { return get_total_size()-1; }
    int     min_own_loc_ind()const { return 0; }
    int     max_own_loc_ind()const { return get_size()-1; }
    bool    check_glob_has_loc_ind(int i_glob)const { return true; }
    bool    check_loc_has_loc_ind(int i_loc)const { return true; }
    bool    is_loc_glob_ind_order_preserv()const { return true; }
};

}  /// namespace communication
}  /// namespace scfd

#endif
