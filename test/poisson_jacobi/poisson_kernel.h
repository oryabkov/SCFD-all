#ifndef __POISSON_KERNEL_H__
#define __POISSON_KERNEL_H__

#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/utils/init_sycl.h>
namespace kernels
{

template<class Mesh,class IdxND,class Array>
struct rhs_init_func
{
    using real = typename Array::value_type;
    using st = scfd::utils::scalar_traits<real>;

    Mesh  mesh;
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

} // namespace kernels

#endif
