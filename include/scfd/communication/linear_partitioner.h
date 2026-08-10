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

#ifndef __SCFD_LINEAR_PARTITIONER_H__
#define __SCFD_LINEAR_PARTITIONER_H__

#include <cmath>
#include <vector>

namespace scfd
{
namespace communication
{

//supposed to satisfy PARTITIONER concept

struct linear_partitioner
{
    int                     total_size;
    std::vector<int>        starts, ends;
    int                     my_rank;
    bool                    is_complete;

    linear_partitioner() {}
    linear_partitioner(const linear_partitioner &part) 
    {
        total_size = part.total_size;
        starts = part.starts;
        ends = part.ends;
        my_rank = part.my_rank;
        is_complete = part.is_complete;
    }
    linear_partitioner(int N, int comm_size, int _my_rank) : is_complete(false)
    {
        int     sz_per_proc = (int)ceil(float(N)/comm_size);
        int     curr = 0;
        for (int iproc = 0;iproc < comm_size;++iproc) {
            starts.push_back(curr);
            curr += sz_per_proc; if (curr > N) curr = N;
            ends.push_back(curr);
        }
        total_size = N;
        my_rank = _my_rank;
    }

    //'read' before construction part:
    int     get_own_rank()const { return my_rank; }
    bool    check_glob_owned(int i)const { return check_glob_owned(i, get_own_rank()); }
    int     get_total_size()const { return total_size; }
    //get_size is not part of PARTITIONER concept
    int     get_size(int rank)const { return ends[rank]-starts[rank]; }
    int     get_size()const { return get_size(get_own_rank()); }
    int     own_glob_ind(int i)const
    {
        return i + starts[get_own_rank()];
    }
    int     own_glob_ind_2_ind(int i_glob)const
    {
        return i_glob - starts[get_own_rank()];
    }

    //'construction' part:
    void    add_stencil_element(int i_glob)
    {
    }
    void    complete()
    {
        is_complete = true;
    }

    //'read' after construction part:
    //here i and rank could be any (from existing range) both before and after construction; this exeeds demands of PARTITIONER concept
    int     get_rank(int i)const
    {
        int     res = 0;
        while (i >= ends[res]) ++res;
        return res;
    }
    bool    check_glob_owned(int i, int rank)const { return ((i >= starts[rank])&&(i < ends[rank])); }
};

}  /// namespace communication
}  /// namespace scfd

#endif
