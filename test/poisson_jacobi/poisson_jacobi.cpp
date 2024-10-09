
#include <iostream>
#include <cmath>
#include "current_dim.h"
#include "current_poisson_solver.h"

static const int dim = current_dim;
using vec_t = poisson_solver_t::vec_type;
using idx_nd_t = poisson_solver_t::idx_nd_type;
using st = scfd::utils::scalar_traits<real>;

int main(int argc, char const *args[])
{
    if (argc != dim+3)
    {
        std::cout << "USAGE: " << args[0] << " nx" << (dim>1?" ny":"") << (dim>2?" nz":"") << 
            " max_iters eps" << std::endl;
        return 1;
    }
    idx_nd_t mesh_sz;
    for (int j = 0;j < dim;++j)
    {
        mesh_sz[j] = std::stoi(args[j+1]);
    }
    int     max_iters = std::stoi(args[dim+1]);
    real    eps = std::stof(args[dim+2]);
    vec_t   dom_sz = vec_t::make_ones()*real(2)*st::pi(),
            wave_numbers = vec_t::make_ones();

    std::cout << "mesh_sz.components_prod() = " << mesh_sz.components_prod() << std::endl;

    poisson_solver_t poisson_solver(mesh_sz, dom_sz);

    poisson_solver.init_rhs(wave_numbers);
    poisson_solver.solve(eps, max_iters);

    auto x_ref_view = poisson_solver.get_x_ref().create_view(true);
    auto x_view = poisson_solver.get_x().create_view(true);
    real l_max_norm = 0.;
    for (int i = 0;i < mesh_sz.components_prod();++i)
    {
        l_max_norm = std::max(l_max_norm, std::abs(x_ref_view.raw_ptr()[i]-x_view.raw_ptr()[i]));
    }
    x_ref_view.release(false); x_view.release(false);

    std::cout << "l_max_norm = " << l_max_norm << std::endl;
    
    return 0;
}