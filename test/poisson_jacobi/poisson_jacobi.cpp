
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include "current_dim.h"
#include "current_poisson_solver.h"

static const int dim = current_dim;
using vec_t          = poisson_solver_t::vec_type;
using idx_nd_t       = poisson_solver_t::idx_nd_type;
using st             = scfd::utils::scalar_traits<real>;

int main( int argc, char const *args[] )
{
    const bool ci     = argc > 1 && std::string( args[1] ) == "--ci";
    const int  offset = ci ? 1 : 0;
    if ( argc != dim + 3 + offset )
    {
        std::cout << "USAGE: " << args[0] << " [--ci] nx" << ( dim > 1 ? " ny" : "" ) << ( dim > 2 ? " nz" : "" )
                  << " max_iters eps" << std::endl;
        return 1;
    }
    idx_nd_t mesh_sz;
    for ( int j = 0; j < dim; ++j )
    {
        mesh_sz[j] = std::stoi( args[j + 1 + offset] );
        if ( mesh_sz[j] < 3 || ( ci && mesh_sz[j] > 65 ) )
        {
            std::cerr << "invalid mesh extent" << std::endl;
            return 2;
        }
    }
    int  max_iters = std::stoi( args[dim + 1 + offset] );
    real eps       = std::stof( args[dim + 2 + offset] );
    if ( max_iters < 1 || ( ci && max_iters > 512 ) || !std::isfinite( eps ) || eps <= real( 0 ) )
    {
        std::cerr << "invalid iteration limit or tolerance" << std::endl;
        return 2;
    }
    vec_t dom_sz = vec_t::make_ones() * real( 2 ) * st::pi(), wave_numbers = vec_t::make_ones();

    std::cout << "mesh_sz.components_prod() = " << mesh_sz.components_prod() << std::endl;

    poisson_solver_t poisson_solver( mesh_sz, dom_sz );

    poisson_solver.init_rhs( wave_numbers );
    const bool converged = poisson_solver.solve( eps, max_iters );

    auto x_ref_view = poisson_solver.get_x_ref().create_view( true );
    auto x_view     = poisson_solver.get_x().create_view( true );
    real l_max_norm = 0.;
    bool finite     = true;
    for ( int i = 0; i < mesh_sz.components_prod(); ++i )
    {
        const real reference = x_ref_view.raw_ptr()[i], value = x_view.raw_ptr()[i];
        finite     = finite && std::isfinite( reference ) && std::isfinite( value );
        l_max_norm = std::max( l_max_norm, std::abs( reference - value ) );
    }
    x_ref_view.release( false );
    x_view.release( false );

    std::cout << "l_max_norm = " << l_max_norm << std::endl;

    if ( ci )
    {
        if ( !converged )
        {
            std::cerr << "CI failure: Jacobi did not converge within the iteration limit" << std::endl;
            return 3;
        }
        // The product of unit-frequency sines is an eigenvector of both the
        // continuous and finite-difference Laplacians. Their reciprocal
        // eigenvalue difference bounds discretization error at every node.
        real discrete_eigenvalue = real( 0 );
        for ( int j = 0; j < dim; ++j )
        {
            const real h     = dom_sz[j] / ( mesh_sz[j] - 1 );
            const real ratio = real( 2 ) * std::sin( h / real( 2 ) ) / h;
            discrete_eigenvalue += ratio * ratio;
        }
        const real discretization = std::abs( real( 1 ) / discrete_eigenvalue - real( 1 ) / dim );
        const real error_limit =
            real( 1.25 ) * discretization + real( 8 ) * eps + real( 128 ) * std::numeric_limits<real>::epsilon();
        std::cout << "CI error limit = " << error_limit << std::endl;
        if ( !finite || !std::isfinite( l_max_norm ) || l_max_norm > error_limit )
        {
            std::cerr << "CI failure: nonfinite solution or excessive numerical error" << std::endl;
            return 4;
        }
        std::cout << "CI Poisson convergence and numerical checks passed" << std::endl;
    }

    return 0;
}
