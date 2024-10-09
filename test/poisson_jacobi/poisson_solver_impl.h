#ifndef __POISSON_SOLVER_IMPL_H__
#define __POISSON_SOLVER_IMPL_H__

#include "poisson_solver.h"

namespace kernels
{

template<class Mesh,class IdxND,class Array>
struct rhs_init_func
{
    using real = typename Array::value_type;
    using st = scfd::utils::scalar_traits<real>;

    Mesh mesh;
    IdxND wave_numbers;
    Array rhs,x_ref;
    __DEVICE_TAG__ void operator()(IdxND idx)const
    {
        real rhs_val = real(1);
        #pragma unroll
        for (int j = 0;j < Mesh::dim;++j)
        {
            rhs_val *= st::sin(real(2)*st::pi()*wave_numbers[j]*mesh.coord(j,idx[j])/mesh.dom_sz()[j]);
        }
        rhs(idx) = rhs_val;

        real ref_mul = real(0);
        #pragma unroll
        for (int j = 0;j < Mesh::dim;++j)
        {
            ref_mul += real(1)/st::sqr(real(2)*st::pi()*wave_numbers[j]/mesh.dom_sz()[j]);
        }
        x_ref(idx) = rhs_val/ref_mul;
    }
};

template<class IdxND,class Array>
struct vanish_func
{
    using real = typename Array::value_type;

    Array a;
    __DEVICE_TAG__ void operator()(IdxND idx)const
    {
        a(idx) = real(0);
    }
};

template<class IdxND,class Array>
struct sqr_func
{
    using real = typename Array::value_type;

    Array a_in, a_out;
    __DEVICE_TAG__ void operator()(IdxND idx)const
    {
        real a_val = a_in(idx);
        a_out(idx) = a_val*a_val;
    }
};

/// This functor calculates residual (pointwise square of it) for input x array as byproduct of iteration calculation
template<class Mesh,class IdxND,class Array>
struct iter_func
{
    using real = typename Array::value_type;
    using st = scfd::utils::scalar_traits<real>;
    
    Mesh mesh;
    Array rhs,x,x_residual,x_new;
    __DEVICE_TAG__ void operator()(IdxND idx)const
    {
        if (mesh.check_is_on_border(idx)) return;

        real num = rhs(idx), den = real(0);
        #pragma unroll
        for (int j = 0;j < Mesh::dim;++j)
        {
            real hj = mesh.step_sz()[j];
            for (int sign = -1;sign <= 1;sign+=2)
            {
                IdxND idx_nb = idx;
                idx_nb[j] += sign;
                num += x(idx_nb)/(hj*hj);
                den += real(1)/(hj*hj);
            }
        }
        x_residual(idx) = st::sqr(num - den*x(idx));
        x_new(idx) = num/den;
    }
};

}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::init_rhs(idx_nd_type wave_numbers)
{
    
    for_each_nd_(kernels::rhs_init_func<mesh_type,idx_nd_type,array_type>{mesh_,wave_numbers,rhs_,x_ref_},mesh_.size());
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::fill_zero(array_type a)
{
    for_each_nd_(kernels::vanish_func<idx_nd_type,array_type>{a},mesh_.size());
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_sum(array_type a)
{
    return reduce_(a.size(), a.raw_ptr(), T(0));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_norm(array_type a)
{
    for_each_nd_(kernels::sqr_func<idx_nd_type,array_type>{a,tmp_},mesh_.size());
    return std::sqrt(calc_sum(tmp_));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::perform_iter()
{
    for_each_nd_(kernels::iter_func<mesh_type,idx_nd_type,array_type>{mesh_,rhs_,x_,x_residual_,x_buf_},mesh_.size());
    return std::sqrt(calc_sum(x_residual_));
}


#endif
