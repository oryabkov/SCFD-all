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
using comm_info_t = communication::mpi_comm_info;
using part_t = communication::rect_partitioner<dim, ordinal, big_ordinal, comm_info_t>;
using idx_t = static_vec::vec<ordinal, dim>;
using periodic_flags_t = static_vec::vec<bool,dim>;
using rect_t = static_vec::rect<ordinal, dim>;
using big_idx_t = static_vec::vec<big_ordinal, dim>;
using big_rect_t = static_vec::rect<big_ordinal, dim>;
using for_each_t = for_each::serial_cpu_nd<dim, ordinal>;
using mem_t = memory::host;
using array_t = arrays::array_nd<value_t, dim, mem_t>;
using dist_t = communication::mpi_rect_distributor<value_t, dim, mem_t, for_each_t, ordinal, big_ordinal>;

int main(int argc, char *args[])
{
    ordinal    stencil = 1;
    ordinal    max_stencil_order = 1;

    comm_t comm(argc, args);
    comm_info_t comm_world = comm.comm_world();
    log_t  log;

    if (comm_world.num_procs != 2)
    {
        std::cout << "only np==2 test case is now implemented" << std::endl;
        return -2;
    }

    big_ordinal size = 10;
    big_idx_t   dom_sz(size, size, size);
    part_t      part(comm_world, dom_sz);
    part.proc_rects = { {{0,0,0}, {size/2,size,size}}, {{size/2,0,0}, {size,size,size}} };
    periodic_flags_t periodic_flags(true, false, false);
    dist_t      dist;

    /* ------------------------ */

    log.info_all("init distributor");
    dist.init(part, periodic_flags, stencil, max_stencil_order);

    big_rect_t  my_own_glob_rect = part.proc_rects[comm_world.myid];
    rect_t      my_own_loc_rect = rect_t(idx_t::make_zero(), my_own_glob_rect.calc_size());
    rect_t      my_loc_rect = my_own_loc_rect;
    my_loc_rect.i1 -= idx_t(stencil, stencil, stencil);
    my_loc_rect.i2 += idx_t(stencil, stencil, stencil);

    /* ------------------------ */

    log.info_all("allocating data array");
    array_t  loc_data_array;
    loc_data_array.init(my_loc_rect.i2-my_loc_rect.i1,my_loc_rect.i1);

    /* ------------------------ */

    log.info_all("filling data array");
    auto data_view1 = loc_data_array.create_view(false);

    for (ordinal ix = my_loc_rect.i1[0];ix < my_loc_rect.i2[0];++ix)
    for (ordinal iy = my_loc_rect.i1[1];iy < my_loc_rect.i2[1];++iy)
    for (ordinal iz = my_loc_rect.i1[2];iz < my_loc_rect.i2[2];++iz)
    {
        data_view1(ix,iy,iz) = static_cast<value_t>(comm_world.myid);
    }

    std::string str1 = "data_view1(:, 0, 0) [";
    for (ordinal ix = my_loc_rect.i1[0];ix < my_loc_rect.i2[0];++ix)
    {
        str1 += std::to_string(data_view1(ix, 5, 5)) + ", ";
        // log.info_all( std::to_string(ix) + ": " + std::to_string(data_view1(ix, 0, 0)) );
    }
    str1 += "]";
    log.info_all(str1);

    data_view1.release(true);

    /* ------------------------ */

    log.info_all("sync array");
    dist.sync(loc_data_array);

    /* ------------------------ */

    log.info_all("check for data");
    auto data_view2 = loc_data_array.create_view(true);

    std::string str2 = "data_view2(:, 0, 0) [";
    for (ordinal ix = my_loc_rect.i1[0];ix < my_loc_rect.i2[0];++ix)
    {
        str2 += std::to_string(data_view2(ix, 5, 5)) + ", ";
        // log.info_all( std::to_string(ix) + ": " + std::to_string(data_view2(ix, 0, 0)) );
    }
    str2 += "]";
    log.info_all(str2);
}
