// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_VEC_H__
#define __SCFD_VEC_H__

#include <type_traits>
#include <scfd/utils/device_tag.h>
#include "detail/bool_array.h"
#include "detail/has_subscript_operator.h"

namespace scfd
{
namespace static_vec 
{

template<class T,int Dim>
class vec
{
    template<class X>using help_t = T;
public:

    T d[Dim];

    typedef T                   value_type;
    static const int            dim = Dim;

    //__DEVICE_TAG__                      vec() {}
    __DEVICE_TAG__                      vec();
    /// ISSUE Still not sure about this static_cast here...
    template<typename... Args,
             class = typename std::enable_if<sizeof...(Args) == Dim>::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< 
                                      std::is_convertible<Args,help_t<Args> >::value... 
                                  >::value
                              >::type>
    __DEVICE_TAG__                      vec(const Args&... args) : d{static_cast<T>(args)...}
    {
    }
    __DEVICE_TAG__                      vec(const vec &v);
    template<class Vec, 
             class = typename std::enable_if<detail::has_subscript_operator<Vec,int>::value>::type>
    __DEVICE_TAG__                      vec(const Vec &v)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] = v[j];
    }

    __DEVICE_TAG__ vec                  operator*(value_type mul)const
    {
        vec res;
        #pragma unroll
        for (int j = 0;j < dim;++j) res.d[j] = d[j]*mul;
        return res;
    }
    __DEVICE_TAG__ vec                  operator/(value_type x)const
    {
        return operator*(value_type(1.)/x);
    }
    __DEVICE_TAG__ vec                  operator+(const vec &x)const
    {
        vec res;
        #pragma unroll
        for (int j = 0;j < dim;++j) res.d[j] = d[j] + x.d[j];
        return res;
    }
    __DEVICE_TAG__ vec                  operator-(const vec &x)const
    {
        vec res;
        #pragma unroll
        for (int j = 0;j < dim;++j) res.d[j] = d[j] - x.d[j];
        return res;
    }
    __DEVICE_TAG__ vec                  operator-()const
    {
        vec res;
        #pragma unroll
        for (int j = 0;j < dim;++j) res.d[j] = -d[j];
        return res;
    }
    __DEVICE_TAG__ vec                  inverted()const
    {
        return -(*this);
    }    
    __DEVICE_TAG__ value_type            &operator[](int j) { return d[j]; }
    __DEVICE_TAG__ const value_type      &operator[](int j)const { return d[j]; }
    __DEVICE_TAG__ value_type            &operator()(int j) { return d[j]; }
    __DEVICE_TAG__ const value_type      &operator()(int j)const { return d[j]; }
    template<int J>
    __DEVICE_TAG__ value_type            &get() { return d[J]; }
    template<int J>
    __DEVICE_TAG__ const value_type      &get()const { return d[J]; }

    __DEVICE_TAG__ value_type            norm2_sq()const
    {
        value_type   res(0.);
        #pragma unroll
        for (int j = 0;j < dim;++j) res += d[j]*d[j];
        return res;
    }
    __DEVICE_TAG__ value_type            norm2()const
    {
        value_type   res(0.);
        #pragma unroll
        for (int j = 0;j < dim;++j) res += d[j]*d[j];
        return sqrt(res);
    }

    static __DEVICE_TAG__ vec            make_zero()
    {
        vec res;
        #pragma unroll
        for (int j = 0;j < dim;++j) res.d[j] = value_type(0.);
        return res;
    }
    
    __DEVICE_TAG__ vec                   &operator=(const vec &v);
    template<class Vec, 
             class = typename std::enable_if<detail::has_subscript_operator<Vec,int>::value>::type>
    __DEVICE_TAG__ vec                   &operator=(const Vec &v)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] = v[j];
        return *this;    
    }
    __DEVICE_TAG__ vec                   &operator+=(const vec &v)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] += v.d[j];
        return *this;
    }
    //TODO check size (statically)
    __DEVICE_TAG__ vec                   &operator-=(const vec &v)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] -= v.d[j];
        return *this;
    }
    __DEVICE_TAG__ vec                   &operator*=(const value_type &mul)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] *= mul;
        return *this;
    }
    __DEVICE_TAG__ vec                   &operator/=(const value_type &mul)
    {
        #pragma unroll
        for (int j = 0;j < dim;++j) d[j] /= mul;
        return *this;
    }
};

template<class T,int Dim>
vec<T,Dim>::vec() = default;

template<class T,int Dim>
vec<T,Dim>::vec(const vec &v) = default;

template<class T,int Dim>
vec<T,Dim>                  &vec<T,Dim>::operator=(const vec &v) = default;

template<class T,int Dim>
__DEVICE_TAG__ vec<T,Dim>   operator*(T mul, const vec<T,Dim> &v)
{
    return v*mul;
}

template<class T,int Dim>
__DEVICE_TAG__ T            scalar_prod(const vec<T,Dim> &v1, const vec<T,Dim> &v2)
{
    T   res(0.);
    #pragma unroll
    for (int j = 0;j < Dim;++j) res += v1[j]*v2[j];
    return res;
}

template<class T,int Dim>
__DEVICE_TAG__ vec<T,Dim>   vector_prod(const vec<T,Dim> &v1, const vec<T,Dim> &v2)
{
    static_assert(Dim==3, "static_vec::vector_prod: trying to apply to non 3d vectors");
    vec<T,Dim>  res;
    res[0] =   v1[1]*v2[2] - v1[2]*v2[1];
    res[1] = -(v1[0]*v2[2] - v1[2]*v2[0]);
    res[2] =   v1[0]*v2[1] - v1[1]*v2[0];
    return res;
}

template<class T,int Dim>
__DEVICE_TAG__ T triple_prod(const vec<T,Dim>& x,
                             const vec<T,Dim>& y,
                             const vec<T,Dim>& z)
{
    static_assert(Dim==3, "static_vec::triple_prod: trying to apply to non 3d vectors");
    return scalar_prod(x,vector_prod(y,z));
}

}

}

#endif
