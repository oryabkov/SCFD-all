#if !defined( PLATFORM_SERIAL_CPU ) || defined( PLATFORM_MPI )
#    error "test_platform_trivial_comm.cpp requires PLATFORM_SERIAL_CPU without PLATFORM_MPI"
#endif

#include <iostream>
#include <string>
#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
#include <scfd/platform/platform.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/rect_distributor.h>


/// Selected SERIAL platform integration test for the trivial communicator.
/// Goal: make trivial_comm/trivial_platform/trivial_message_queue work so that this
/// runs WITHOUT mpiexec and reproduces a periodic (x) halo exchange done as a self-send.
using namespace scfd;

using platform_t     = platform::current<>;
using ordinal        = platform_t::ordinal_type;
using big_ordinal    = platform_t::big_ordinal_type;
using value_t        = unsigned int;
static const int dim = 3;

using mem_t         = platform_t::memory_type;
using environment_t = platform_t::communication_environment_type;
using comm_info_t   = platform_t::communicator_type;
using part_t        = communication::rect_partitioner<dim, ordinal, big_ordinal, comm_info_t>;
using for_each_t    = platform_t::for_each_nd_type<dim>;

using idx_t            = static_vec::vec<ordinal, dim>;
using periodic_flags_t = static_vec::vec<bool, dim>;
using rect_t           = static_vec::rect<ordinal, dim>;
using big_idx_t        = static_vec::vec<big_ordinal, dim>;
using big_rect_t       = static_vec::rect<big_ordinal, dim>;
using array_t          = arrays::array_nd<value_t, dim, mem_t>;
using dist_t = communication::rect_distributor<value_t, dim, mem_t, for_each_t, ordinal, big_ordinal, comm_info_t>;

int main( int argc, char *args[] )
{
    ordinal stencil           = 1;
    int     max_stencil_order = 1;

    environment_t environment( argc, args );
    comm_info_t   comm_world = environment.comm_world();
    platform_t::init( comm_world );

    const comm_info_t copied_comm = comm_world;
    if ( comm_world.num_procs != 1 || comm_world.myid != 0 || comm_world.queue == nullptr ||
         copied_comm.queue != comm_world.queue || environment.comm_world().queue != comm_world.queue )
    {
        std::cerr << "FAILED: invalid trivial communicator or shared queue" << std::endl;
        return 1;
    }

    big_ordinal size = 10;
    big_idx_t   dom_sz( size, size, size );
    part_t      part( comm_world, dom_sz );
    // single process owns the whole domain (no decomposition)
    part.proc_rects = { { { 0, 0, 0 }, { size, size, size } } };
    periodic_flags_t periodic_flags( true, false, false ); // periodic in x => self-exchange
    dist_t           dist;

    /* ------------------------ */

    std::cout << "init distributor" << std::endl;
    dist.init( part, periodic_flags, stencil, max_stencil_order );

    big_rect_t my_own_glob_rect = part.proc_rects[comm_world.myid];
    rect_t     my_own_loc_rect  = rect_t( idx_t::make_zero(), my_own_glob_rect.calc_size() );
    rect_t     my_loc_rect      = my_own_loc_rect;
    my_loc_rect.i1 -= idx_t( stencil, stencil, stencil );
    my_loc_rect.i2 += idx_t( stencil, stencil, stencil );

    /* ------------------------ */

    std::cout << "allocating data array" << std::endl;
    array_t loc_data_array;
    loc_data_array.init( my_loc_rect.i2 - my_loc_rect.i1, my_loc_rect.i1 );

    /* ------------------------ */

    std::cout << "filling data array" << std::endl;
    auto data_view1 = loc_data_array.create_view( false );

    // interior cells get their global x index; x-ghost cells get a sentinel so we can
    // see them change after the sync.
    const value_t ghost_sentinel = 999;
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
        for ( ordinal iy = my_loc_rect.i1[1]; iy < my_loc_rect.i2[1]; ++iy )
            for ( ordinal iz = my_loc_rect.i1[2]; iz < my_loc_rect.i2[2]; ++iz )
            {
                bool interior_x          = ( ix >= my_own_loc_rect.i1[0] && ix < my_own_loc_rect.i2[0] );
                data_view1( ix, iy, iz ) = interior_x ? static_cast<value_t>( ix ) : ghost_sentinel;
            }

    std::string str1 = "before sync, (:,5,5) [";
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
        str1 += std::to_string( data_view1( ix, 5, 5 ) ) + ", ";
    str1 += "]";
    std::cout << str1 << std::endl;

    data_view1.release( true );

    /* ------------------------ */

    std::cout << "sync array" << std::endl;
    dist.sync( loc_data_array );

    /* ------------------------ */

    std::cout << "check for data" << std::endl;
    auto data_view2 = loc_data_array.create_view( true );

    std::string str2 = "after  sync, (:,5,5) [";
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
        str2 += std::to_string( data_view2( ix, 5, 5 ) ) + ", ";
    str2 += "]";
    std::cout << str2 << std::endl;

    std::cout << "expected     [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, ] (periodic wrap in x)" << std::endl;

    bool is_failed = false;
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
    {
        const value_t expected =
            ix < 0 ? static_cast<value_t>( size - 1 ) : ( ix >= size ? 0 : static_cast<value_t>( ix ) );
        if ( data_view2( ix, 5, 5 ) != expected )
        {
            std::cerr << "FAILED at x=" << ix << ": got " << data_view2( ix, 5, 5 ) << ", expected " << expected
                      << std::endl;
            is_failed = true;
        }
    }
    data_view2.release( false );

    if ( is_failed )
        return 1;

    std::cout << "PASSED" << std::endl;
    return 0;
}
