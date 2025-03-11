#ifndef __POISSON_SOLVER_IMPL_H__
#define __POISSON_SOLVER_IMPL_H__

#include "poisson_solver.h"

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::init_rhs(idx_nd_type wave_numbers)
{

    for_each_nd_(rhs_init_func_t{mesh_,wave_numbers,rhs_,x_ref_},mesh_.size());
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
void poisson_solver<T,ForEachND,Reduce,Memory,Dim>::fill_zero(array_type a)
{
    for_each_nd_(vanish_func_t{a},mesh_.size());
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_sum(array_type a)
{
    return reduce_(a.size(), a.raw_ptr(), T(0));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::calc_norm(array_type a)
{
    for_each_nd_(sqr_func_t{a,tmp_},mesh_.size());
    return std::sqrt(calc_sum(tmp_));
}

template<class T,class ForEachND,class Reduce,class Memory,int Dim>
T   poisson_solver<T,ForEachND,Reduce,Memory,Dim>::perform_iter()
{
    for_each_nd_(iter_func_t{mesh_,rhs_,x_,x_residual_,x_buf_},mesh_.size());
    return std::sqrt(calc_sum(x_residual_));
}


#endif
