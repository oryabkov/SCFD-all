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

#ifndef __SCFD_STATIC_VEC_VEC_DIV_SCALAR_H__
#define __SCFD_STATIC_VEC_VEC_DIV_SCALAR_H__

namespace scfd
{
namespace static_vec 
{

template<class T,int Dim>
class vec;

namespace detail
{

template<class T,int Dim,typename std::enable_if<!std::is_floating_point<T>::value,bool>::type = true>
inline __DEVICE_TAG__ vec<T,Dim> vec_div_scalar(const vec<T,Dim> &v, T div)
{
    vec<T,Dim> res;
    #pragma unroll
    for (int j = 0;j < Dim;++j) res.d[j] = v.d[j]/div;
    return res;
}

template<class T,int Dim,typename std::enable_if<std::is_floating_point<T>::value,bool>::type = true>
inline __DEVICE_TAG__ vec<T,Dim> vec_div_scalar(const vec<T,Dim> &v, T div)
{
    return v.operator*(T(1)/div);
}

template<class T,int Dim,typename std::enable_if<!std::is_floating_point<T>::value,bool>::type = true>
inline __DEVICE_TAG__ void vec_div_scalar_inplace(T div, vec<T,Dim> &v)
{
    #pragma unroll
    for (int j = 0;j < Dim;++j) v.d[j] /= div;
}

template<class T,int Dim,typename std::enable_if<std::is_floating_point<T>::value,bool>::type = true>
inline __DEVICE_TAG__ void vec_div_scalar_inplace(T div, vec<T,Dim> &v)
{
    v.operator*=(T(1)/div);
}


}

}
}

#endif
