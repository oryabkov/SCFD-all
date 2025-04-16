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

#include <iostream>
#include <scfd/utils/log_mpi.h>
#include <scfd/memory/host.h>
#include <scfd/memory/cuda.h>
#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
#include <scfd/arrays/array_nd.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/mpi_rect_distributor.h>

/// Just for test!!
using namespace scfd;

using log_t = utils::log_mpi;
using ordinal = int;
using big_ordinal = long int;
using value_t = unsigned int;
static const int dim = 3;
using comm_t = communication::mpi_wrap;
using part_t = communication::rect_partitioner<dim,ordinal,big_ordinal>;
using idx_t = static_vec::vec<ordinal,dim>;
using periodic_flags_t = static_vec::vec<bool,dim>;
using rect_t = static_vec::rect<ordinal,dim>;
using big_idx_t = static_vec::vec<big_ordinal,dim>;
using big_rect_t = static_vec::rect<big_ordinal,dim>;
using for_each_t = for_each::serial_cpu_nd<dim,ordinal>;
using mem_t = memory::host;
using array_t = arrays::array_nd<value_t,dim,mem_t>;
using dist_t = communication::mpi_rect_distributor<value_t,dim,mem_t,for_each_t,ordinal,big_ordinal>;

big_idx_t periodic_rem(periodic_flags_t periodic_flags, big_idx_t dom_sz, big_idx_t idx)
{
    big_idx_t res = idx;
    for (int j = 0;j < dim;++j)
    {
        if (!periodic_flags[j]) continue;
        while (res[j] < 0) res[j] += dom_sz[j];
        while (res[j] >= dom_sz[j]) res[j] -= dom_sz[j];
    }
    return res;
}

value_t ref_value(periodic_flags_t periodic_flags, big_idx_t dom_sz, big_idx_t idx)
{
    idx = periodic_rem(periodic_flags,dom_sz,idx);
    if (big_rect_t(dom_sz).is_own(idx))
    {
        return 
            static_cast<value_t>(
                idx[0]*dom_sz[1]*dom_sz[2] + idx[1]*dom_sz[2] + idx[2]
            );
    }
    return static_cast<value_t>(-1); //TODO use type traits max_value
}

int stencil_order(big_rect_t  my_own_glob_rect, big_idx_t idx)
{
    int res = 0;
    for (int j = 0;j < dim;++j)
    {
        if (!((idx[j]>=my_own_glob_rect.i1[j])&&(idx[j]<my_own_glob_rect.i2[j]))) res += 1;
    }
    return res;
}

int main(int argc, char *args[])
{
    if (argc < 6)
    {
        std::cout << "USAGE: " << std::string(args[0]) << " stencil max_stencil_order periodic_flag_x periodic_flag_y periodic_flag_z" << std::endl;
        return -1;
    }
    ordinal     stencil = std::stoi(args[1]);
    int         max_stencil_order = std::stoi(args[2]);
    int         periodic_flag_x = std::stoi(args[3]);
    int         periodic_flag_y = std::stoi(args[4]);
    int         periodic_flag_z = std::stoi(args[5]);

    comm_t comm(argc, args);
    log_t  log;

    if (comm.data.num_procs != 3)
    {
        std::cout << "only np==3 test case is now implemented" << std::endl;
        return -2;
    }

    big_idx_t   dom_sz(100,100,100);
    part_t      part(comm.data, dom_sz);
    part.proc_rects = {{{0,0,0},{100,50,100}},{{0,50,0},{100,100,50}},{{0,50,50},{100,100,100}}};
    periodic_flags_t periodic_flags(periodic_flag_x,periodic_flag_y,periodic_flag_z);
    dist_t      dist;
    log.info_all("init distributor");
    dist.init(part,periodic_flags,stencil,max_stencil_order);

    big_rect_t  my_own_glob_rect = part.proc_rects[comm.data.myid];
    rect_t      my_own_loc_rect = rect_t(idx_t::make_zero(), my_own_glob_rect.calc_size());
    rect_t      my_loc_rect = my_own_loc_rect;
    my_loc_rect.i1 -= idx_t(stencil,stencil,stencil);
    my_loc_rect.i2 += idx_t(stencil,stencil,stencil);
    
    //std::cout << "only np==3 test case is now implemented" << std::endl;
    log.info_all("allocating data array");
    array_t  loc_data_array;
    loc_data_array.init(my_loc_rect.i2-my_loc_rect.i1,my_loc_rect.i1);

    log.info_all("filling data array");
    auto data_view1 = loc_data_array.create_view(false);
    //first fill all the data including halo with -1 (as outer nonperiodic regions wont sync)
    for (ordinal ix = my_loc_rect.i1[0];ix < my_loc_rect.i2[0];++ix)
    for (ordinal iy = my_loc_rect.i1[1];iy < my_loc_rect.i2[1];++iy)
    for (ordinal iz = my_loc_rect.i1[2];iz < my_loc_rect.i2[2];++iz)
    {
        data_view1(ix,iy,iz) = static_cast<value_t>(-1);
    }
    //fill own data using global linear index with mostly handwritten functions
    for (ordinal ix = my_own_loc_rect.i1[0];ix < my_own_loc_rect.i2[0];++ix)
    for (ordinal iy = my_own_loc_rect.i1[1];iy < my_own_loc_rect.i2[1];++iy)
    for (ordinal iz = my_own_loc_rect.i1[2];iz < my_own_loc_rect.i2[2];++iz)
    {
        big_idx_t idx = big_idx_t(ix,iy,iz) + my_own_glob_rect.i1;
        data_view1(ix,iy,iz) = ref_value(periodic_flags, dom_sz, idx);
    }
    data_view1.release(true);

    log.info_all("sync array");
    dist.sync(loc_data_array);

    log.info_all("check for data");
    auto data_view2 = loc_data_array.create_view(true);
    int error_flag = 0;
    for (ordinal ix = my_loc_rect.i1[0];ix < my_loc_rect.i2[0];++ix)
    for (ordinal iy = my_loc_rect.i1[1];iy < my_loc_rect.i2[1];++iy)
    for (ordinal iz = my_loc_rect.i1[2];iz < my_loc_rect.i2[2];++iz)
    {
        big_idx_t idx = big_idx_t(ix,iy,iz) + my_own_glob_rect.i1;
        if (stencil_order(my_own_glob_rect, idx) > max_stencil_order) continue;
        auto expected_val = ref_value(periodic_flags, dom_sz, idx);
        if (data_view2(ix,iy,iz) != expected_val)
        {
            log.error_f("value mistmatch at index %d,%d,%d : expected %u got %u",ix,iy,iz,expected_val,data_view2(ix,iy,iz));
            error_flag = 1;
        }
    }
    data_view2.release(false);

    if (error_flag == 0)
    {
        log.info_all("PASSED");
    }
    else
    {
        log.info_all_f("error_flag = %d", error_flag);
    }

    return error_flag;
}