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

#ifndef __SCFD_MAT_H__
#define __SCFD_MAT_H__

#include <type_traits>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include "detail/bool_array.h"

namespace scfd
{
namespace static_mat 
{

template<class T,int Dim1,int Dim2>
class mat
{
    template<class X>using help_t = T;
public:

    T d[Dim1][Dim2];

    typedef T                   value_type;
    static const int            dim1 = Dim1;
    static const int            dim2 = Dim2;

    __DEVICE_TAG__                      mat();
    __DEVICE_TAG__                      mat(const mat &v);
    template<typename... Args,
             class = typename std::enable_if<sizeof...(Args) == Dim1*Dim2>::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< 
                                      std::is_convertible<Args,help_t<Args> >::value... 
                                  >::value
                              >::type>
    __DEVICE_TAG__                      mat(const Args&... args) : d{static_cast<T>(args)...}
    {
    }

    __DEVICE_TAG__ mat                  operator*(value_type mul)const
    {
        mat res;
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                res.d[i][j] = d[i][j]*mul;
        }
        return res;
    }
    __DEVICE_TAG__ mat                  operator/(value_type x)const
    {
        return operator*(value_type(1)/x);
    }
    __DEVICE_TAG__ mat                  operator+(const mat &x)const
    {
        mat res;
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                res.d[i][j] = d[i][j] + x.d[i][j];
        }
        return res;
    }
    __DEVICE_TAG__ mat                  operator-(const mat &x)const
    {
        mat res;
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                res.d[i][j] = d[i][j] - x.d[i][j];
        }
        return res;
    }
    __DEVICE_TAG__ value_type            &operator()(int i,int j) { return d[i][j]; }
    __DEVICE_TAG__ const value_type      &operator()(int i,int j)const { return d[i][j]; }
    template<int I,int J>
    __DEVICE_TAG__ value_type            &get() { return d[I][J]; }
    template<int I,int J>
    __DEVICE_TAG__ const value_type      &get()const { return d[I][J]; }

    static __DEVICE_TAG__ mat            make_zero()
    {
        mat res;
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                res.d[i][j] = value_type(0);
        }
        return res;
    }
    
    __DEVICE_TAG__ mat                   &operator=(const mat &v);
    __DEVICE_TAG__ mat                   &operator+=(const mat &v)
    {
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                d[i][j] += v.d[i][j];
        }
        return *this;
    }
    //TODO check size (statically)
    __DEVICE_TAG__ mat                   &operator-=(const mat &v)
    {
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                d[i][j] -= v.d[i][j];
        }
        return *this;
    }
    __DEVICE_TAG__ mat                   &operator*=(const value_type &mul)
    {
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                d[i][j] *= mul;
        }
        return *this;
    }
    __DEVICE_TAG__ mat                   &operator/=(const value_type &mul)
    {
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                d[i][j] /= mul;
        }
        return *this;
    }

    __DEVICE_TAG__ static_vec::vec<T,Dim1> operator*(const static_vec::vec<T,Dim2> &v)const
    {
        static_vec::vec<T,Dim1> res;

        #pragma unroll
        for (int i = 0;i < Dim1;++i) {
            T sum = T(0);
            #pragma unroll
            for (int l = 0; l < Dim2; ++l) {
                sum += d[i][l]*v[l];
            }
            res[i] = sum;
        }

        return res;
    }

    template<int Dim3>
    __DEVICE_TAG__ mat<T,Dim1,Dim3>      operator*(const mat<T,Dim2,Dim3> &m)const
    {
        mat<T,Dim1,Dim3> res;

        #pragma unroll
        for (int i = 0;i < Dim1;++i) {
            #pragma unroll
            for (int j = 0;j < Dim3;++j) {
                T sum = T(0);
                #pragma unroll
                for (int l = 0; l < Dim2; ++l) {
                    sum += d[i][l]*m.d[l][j];
                }
                res.d[i][j] = sum;
            }
        }

        return res;
    }
    __DEVICE_TAG__ mat<T,Dim2,Dim1>      transposed()const
    {
        mat<T,Dim2,Dim1> res;
        #pragma unroll
        for (int i = 0;i < dim1;++i) {
            #pragma unroll
            for (int j = 0;j < dim2;++j)
                res.d[j][i] = d[i][j];
        }
        return res;
    }
    __DEVICE_TAG__ T                     trace()const
    {
        T res(0);
        #pragma unroll
        for (int i = 0;i < (dim1 < dim2?dim1:dim2);++i) {
            res += d[i][i];
        }
        return res;
    }
};

template<class T,int Dim1,int Dim2>
mat<T,Dim1,Dim2>::mat() = default;

template<class T,int Dim1,int Dim2>
mat<T,Dim1,Dim2>::mat(const mat &v) = default;

template<class T,int Dim1,int Dim2>
mat<T,Dim1,Dim2>                &mat<T,Dim1,Dim2>::operator=(const mat &v) = default;


template<class T,int Dim1,int Dim2>
__DEVICE_TAG__ mat<T,Dim1,Dim2> operator*(T mul, const mat<T,Dim1,Dim2> &m)
{
    return m*mul;
}

}
}

#endif
