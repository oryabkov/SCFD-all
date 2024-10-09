#ifndef __POISSON_SOLVER_LAMBDA_IMPL_H__
#define __POISSON_SOLVER_LAMBDA_IMPL_H__

#include "poisson_solver.h"

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::init_rhs(idx_nd_type wave_numbers)
{
    ///WARNING we need this strange thing otherwise cuda version fails with segfault (it seems it captures this pointer)
    auto mesh = mesh_;
    auto rhs = rhs_, x_ref = x_ref_;
    for_each_nd_(
        [=] __DEVICE_TAG__ (idx_nd_type idx)
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
        },
        mesh_.size()
    );
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::fill_zero(array_type a)
{
    for_each_nd_(
        [=] __DEVICE_TAG__ (idx_nd_type idx)
        {
            a(idx) = real(0);
        },
        mesh_.size()
    );
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_sum(array_type a)
{
    return reduce_(a.size(), a.raw_ptr(), T(0));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_norm(array_type a)
{
    auto a_sq = tmp_;
    for_each_nd_(
        [=] __DEVICE_TAG__ (idx_nd_type idx)
        {
            real a_val = a(idx);
            a_sq(idx) = a_val*a_val;
        },
        mesh_.size()
    );
    return std::sqrt(calc_sum(a_sq));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::perform_iter()
{
    auto mesh = mesh_;
    auto rhs = rhs_, x = x_, x_residual = x_residual_, x_new = x_buf_;
    /// This functor calculates residual (pointwise square of it) for input x array as byproduct of iteration calculation
    for_each_nd_(
        [=] __DEVICE_TAG__ (idx_nd_type idx)
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
            x_residual(idx) = st::sqr(num - den*x(idx));
            x_new(idx) = num/den;
        },
        mesh_.size()
    );
    return std::sqrt(calc_sum(x_residual_));
}


#endif
