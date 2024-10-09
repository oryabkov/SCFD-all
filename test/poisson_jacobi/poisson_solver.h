#ifndef __POISSON_SOLVER_H__
#define __POISSON_SOLVER_H__

#include <iostream>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/arrays/array_nd.h>

template<class T,int Dim = 3>
struct mesh
{
    static const int dim = Dim;
    using ordinal = int;
    using real = T;
    using idx_nd_type = scfd::static_vec::vec<ordinal,Dim>;
    using vec_type = scfd::static_vec::vec<real,Dim>;
public:
    mesh(idx_nd_type size, vec_type dom_sz) : size_(size),dom_sz_(dom_sz)
    {
        for (int j = 0;j < Dim;++j)
        {
            /// Assume here that mesh is nodal and first and last points are on the border of domain
            step_sz_[j] = dom_sz_[j]/(size_[j]-1);
        }
    }

    __DEVICE_TAG__ const idx_nd_type &size()const
    {
        return size_;
    }
    __DEVICE_TAG__ const vec_type &dom_sz()const
    {
        return dom_sz_;
    }
    __DEVICE_TAG__ const vec_type &step_sz()const
    {
        return step_sz_;
    }
    /// j is direction and i is mesh point number in this direction
    __DEVICE_TAG__ real coord(int j,ordinal i)const
    {
        return step_sz_[j]*i;
    }
    __DEVICE_TAG__ bool check_is_on_border(idx_nd_type idx)const
    {
        bool res = false;
        #pragma unroll
        for (int j = 0;j < Dim;++j)
        {
            if ((idx[j]==0)||(idx[j]==size_[j]-1))
            {
                res = true;
            }
        }
        return res;
    }

private:
    idx_nd_type size_;
    vec_type dom_sz_;
    vec_type step_sz_;
};

/**
 * Solver for model equation:
 * -\Delta u = f
 * on rectangular domain in \mathbf{R}^Dim space.
 * Currently with model rhs made of sin function multiplication (see init_rhs).
 * Currently only zero Dirichlet bc are used.
 **/
template<class T,class ForEachND,class Reduce,class Memory,int Dim = 3>
class poisson_solver
{
public:
    static const int dim = Dim;
    using ordinal = int;
    using real = T;
    using idx_nd_type = scfd::static_vec::vec<ordinal,Dim>;
    using vec_type = scfd::static_vec::vec<real,Dim>;
    using memory_type = Memory;
    using for_each_nd_type = ForEachND;
    using reduce_type = Reduce;
    using mesh_type = mesh<T,Dim>;
    using array_type = scfd::arrays::array_nd<T,dim,Memory>;
    using st = scfd::utils::scalar_traits<T>;
public:
    poisson_solver(idx_nd_type size, vec_type dom_sz) : 
      mesh_(size, dom_sz),
      rhs_(size), x_(size), x_residual_(size), x_buf_(size), x_ref_(size), tmp_(size)
    {
    }
    /// Remove copy to protect from internal arrays loosing
    poisson_solver(const poisson_solver&) = delete;
    poisson_solver &operator=(const poisson_solver&) = delete;

    const array_type &get_x()const
    {
        return x_;
    }
    const array_type &get_x_ref()const
    {
        return x_ref_;
    }

    /// Just as example rhs: sin(2*pi*wave_numbers[0]*x[0]/dom_sz[0])*sin...
    void init_rhs(idx_nd_type wave_numbers);
    /*{
        struct rhs_init_func
        {
            mesh_type mesh;
            idx_nd_type wave_numbers;
            array_type rhs,x_ref;
            __DEVICE_TAG__ void operator()(idx_nd_type idx)const
            {
                real rhs_val = real(1);
                #pragma unroll
                for (int j = 0;j < dim;++j)
                {
                    rhs_val *= st::sin(real(2)*st::pi()*wave_numbers[j]*mesh.coord(j,idx[j])/mesh.dom_sz()[j]);
                }
                rhs(idx) = rhs_val;

                real ref_mul = real(0);
                #pragma unroll
                for (int j = 0;j < dim;++j)
                {
                    ref_mul += real(1)/st::sqr(real(2)*st::pi()*wave_numbers[j]/mesh.dom_sz()[j]);
                }
                x_ref(idx) = rhs_val/ref_mul;
            }
        };
        for_each_nd_(rhs_init_func{mesh_,wave_numbers,rhs_,x_ref_},mesh_.size());
    }*/
    bool solve(real eps, int max_iters)
    {
        /// We need to vanish x_buf_ because of bc
        fill_zero(x_); fill_zero(x_buf_); fill_zero(x_residual_);
        T rhs_norm = calc_norm(rhs_);
        std::cout << "poisson_solver::solve: rhs_norm_2 = " << rhs_norm << std::endl;
        for (int i = 0;i < max_iters;++i)
        {
            T resudial_norm = perform_iter();
            std::cout << "poisson_solver::solve: performed " << i << " iterations" << std::endl;
            std::cout << "poisson_solver::solve: resudial_norm_2 = " << resudial_norm << std::endl;
            std::cout << "poisson_solver::solve: relative resudial norm = " << resudial_norm/rhs_norm << std::endl;
            if (resudial_norm/rhs_norm <= eps) 
            {
                std::cout << "poisson_solver::solve: target relative resudial norm " << eps << " reached; stop iterations" << std::endl;
                return true;
            }
            std::swap(x_,x_buf_);
        }
        return false;
    }

private:
    for_each_nd_type for_each_nd_;
    reduce_type reduce_;
    mesh_type mesh_;
    array_type rhs_, x_, x_residual_, x_buf_, x_ref_, tmp_;

private:
    void fill_zero(array_type a);
    T    calc_sum(array_type a);
    T    calc_norm(array_type a);
    /*{
        struct vanish_func
        {
            array_type a;
            __DEVICE_TAG__ void operator()(idx_nd_type idx)const
            {
                a(idx) = real(0);
            }
        };
        for_each_nd_(vanish_func{a},mesh_.size());
    }*/
    /// takes x_ as input calculates next iteration and puts it into x_buf_ 
    /// as byproduct calculates residual for x_ and returns its norm-2
    T   perform_iter();
    /*{
        struct iter_func
        {
            mesh_type mesh;
            array_type rhs,x,x_new;
            __DEVICE_TAG__ void operator()(idx_nd_type idx)const
            {
                if (mesh.check_is_on_border(idx)) return;

                real num = rhs(idx), den = real(0);
                #pragma unroll
                for (int j = 0;j < dim;++j)
                {
                    real hj = mesh.step_sz()[j];
                    for (int sign = -1;sign <= 1;sign+=2)
                    {
                        idx_nd_type idx_nb = idx;
                        idx_nb[j] += sign;
                        num += x(idx_nb)/(hj*hj);
                        den += real(1)/(hj*hj);
                    }
                }
                x_new(idx) = num/den;
            }
        };
        for_each_nd_(iter_func{mesh_,rhs_,x_,x_buf_},mesh_.size());
        std::swap(x_,x_buf_);
    }*/

};

#endif
