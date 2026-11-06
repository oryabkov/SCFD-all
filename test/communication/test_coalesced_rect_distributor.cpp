#include <iostream>
#include <string>
#include <scfd/memory/host.h>
#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/rect_distributor.h>
#include <scfd/communication/coalesced_rect_distributor.h>
#include <scfd/communication/trivial_comm.h>
#include <scfd/communication/trivial_platform.h>

/// Correctness check for coalesced_rect_distributor: syncs two identically initialized
/// arrays -- one with the old (reference) rect_distributor, one with the new
/// coalesced_rect_distributor -- and asserts the results are bit-identical.
/// trivial_comm only simulates a single rank, so this only exercises the self-to-self
/// (num_procs == 1) coalescing path; real multi-rank correctness is covered separately
/// by test_mpi_coalesced_rect_distributor.cu.
using namespace scfd;

using ordinal         = int;
using big_ordinal     = long int;
using value_t         = long int;
static const int dim  = 3;

using mem_t       = memory::host;
using comm_t      = communication::trivial_platform<mem_t>;
using comm_info_t = communication::trivial_comm<mem_t>;
using part_t      = communication::rect_partitioner<dim, ordinal, big_ordinal, comm_info_t>;
using for_each_t  = for_each::serial_cpu_nd<dim, ordinal>;

using idx_t            = static_vec::vec<ordinal, dim>;
using periodic_flags_t = static_vec::vec<bool, dim>;
using rect_t           = static_vec::rect<ordinal, dim>;
using big_idx_t        = static_vec::vec<big_ordinal, dim>;

using old_dist_t = communication::rect_distributor<value_t, dim, mem_t, for_each_t, ordinal, big_ordinal, comm_info_t>;
using new_dist_t =
    communication::coalesced_rect_distributor<value_t, dim, mem_t, for_each_t, ordinal, big_ordinal, comm_info_t>;

template <int TensorDim>
using array_t = arrays::tensor1_array_nd<value_t, dim, mem_t, TensorDim>;

value_t make_value( ordinal ix, ordinal iy, ordinal iz, int t )
{
    return static_cast<value_t>( ( ix * 1000003LL + iy * 1009LL + iz * 31LL ) * 17LL + t );
}

template <int TensorDim>
int run_case(
    const std::string &name, ordinal stencil, int max_stencil_order, periodic_flags_t periodic_flags,
    big_ordinal size
)
{
    std::cout << "case " << name << " ... " << std::flush;

    comm_t      comm( 0, nullptr );
    comm_info_t comm_world = comm.comm_world();

    big_idx_t dom_sz( size, size, size );
    part_t    part( comm_world, dom_sz );

    old_dist_t dist_old;
    new_dist_t dist_new;
    dist_old.init_for_tensors( TensorDim, part, periodic_flags, stencil, max_stencil_order );
    dist_new.init_for_tensors( TensorDim, part, periodic_flags, stencil, max_stencil_order );

    rect_t my_own_loc_rect = part.get_own_loc_rect();
    rect_t my_loc_rect     = my_own_loc_rect;
    my_loc_rect.i1 -= idx_t( stencil, stencil, stencil );
    my_loc_rect.i2 += idx_t( stencil, stencil, stencil );

    array_t<TensorDim> arr_old, arr_new;
    arr_old.init( my_loc_rect.i2 - my_loc_rect.i1, my_loc_rect.i1 );
    arr_new.init( my_loc_rect.i2 - my_loc_rect.i1, my_loc_rect.i1 );

    const value_t ghost_sentinel = -777;
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
        for ( ordinal iy = my_loc_rect.i1[1]; iy < my_loc_rect.i2[1]; ++iy )
            for ( ordinal iz = my_loc_rect.i1[2]; iz < my_loc_rect.i2[2]; ++iz )
            {
                bool interior = my_own_loc_rect.is_own( idx_t( ix, iy, iz ) );
                for ( int t = 0; t < TensorDim; ++t )
                {
                    value_t v            = interior ? make_value( ix, iy, iz, t ) : ghost_sentinel;
                    arr_old( ix, iy, iz, t ) = v;
                    arr_new( ix, iy, iz, t ) = v;
                }
            }

    dist_old.sync( arr_old );
    dist_new.sync( arr_new );

    int errors_num = 0;
    for ( ordinal ix = my_loc_rect.i1[0]; ix < my_loc_rect.i2[0]; ++ix )
        for ( ordinal iy = my_loc_rect.i1[1]; iy < my_loc_rect.i2[1]; ++iy )
            for ( ordinal iz = my_loc_rect.i1[2]; iz < my_loc_rect.i2[2]; ++iz )
            {
                for ( int t = 0; t < TensorDim; ++t )
                {
                    value_t v_old = arr_old( ix, iy, iz, t );
                    value_t v_new = arr_new( ix, iy, iz, t );
                    if ( v_old != v_new )
                    {
                        if ( errors_num < 10 )
                        {
                            std::cout << std::endl
                                       << "  mismatch at (" << ix << "," << iy << "," << iz << ",t=" << t
                                       << "): old=" << v_old << " new=" << v_new;
                        }
                        ++errors_num;
                    }
                }
            }

    if ( errors_num == 0 )
    {
        std::cout << "PASSED" << std::endl;
    }
    else
    {
        std::cout << std::endl << "FAILED: " << errors_num << " mismatched values" << std::endl;
    }

    return errors_num;
}

int main()
{
    int errors_num = 0;

    errors_num += run_case<1>(
        "periodic_full_order3_stencil2_tensor1", 2, dim, periodic_flags_t( true, true, true ), 10
    );
    errors_num += run_case<2>(
        "periodic_full_order3_stencil2_tensor2", 2, dim, periodic_flags_t( true, true, true ), 10
    );
    errors_num += run_case<2>(
        "order1_stencil1_mixed_periodic", 1, 1, periodic_flags_t( true, false, true ), 10
    );
    errors_num += run_case<1>(
        "order1_stencil2_mixed_periodic", 2, 1, periodic_flags_t( false, true, true ), 10
    );

    if ( errors_num == 0 )
    {
        std::cout << "TEST PASSED" << std::endl;
    }
    else
    {
        std::cout << "TEST FAILED: " << errors_num << " total mismatched values" << std::endl;
    }

    return errors_num == 0 ? 0 : 1;
}
