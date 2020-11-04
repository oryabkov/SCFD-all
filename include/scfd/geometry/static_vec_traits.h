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

#ifndef __SCFD_GEOMETRY_STATIC_VEC_TRAITS_H__
#define __SCFD_GEOMETRY_STATIC_VEC_TRAITS_H__

#include <type_traits>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include "detail/bool_array.h"

/// TODO not sure about 'scalar' vs 'value' concepts
/// i.e. static_vec::vec uses concept of 'value' meaning 
/// that ordinal type also could be used. Here - i'm not sure.

namespace scfd
{
namespace geometry
{

template<class Vec>
struct static_vec_traits
{
    static constexpr int dim = 1;
    typedef Vec vec_type;
    typedef float scalar_type;

    template<int J>
    inline __DEVICE_TAG__ static scalar_type &comp(Vec &v1);
    template<int J>
    inline __DEVICE_TAG__ static const scalar_type &comp(const Vec &v1);

    /// Only if corresponding component exists (according to actual dimension)
    inline __DEVICE_TAG__ static scalar_type &x(Vec &v1);
    inline __DEVICE_TAG__ static scalar_type &y(Vec &v1);
    inline __DEVICE_TAG__ static scalar_type &z(Vec &v1);
    inline __DEVICE_TAG__ static const scalar_type &x(const Vec &v1);
    inline __DEVICE_TAG__ static const scalar_type &y(const Vec &v1);
    inline __DEVICE_TAG__ static const scalar_type &z(const Vec &v1);

    inline __DEVICE_TAG__ static Vec zero();

    /// v2  = v1
    inline __DEVICE_TAG__ static void assign(const Vec &v1, Vec &v2);
    /// v2  += v1
    inline __DEVICE_TAG__ static void add(const Vec &v1, Vec &v2);
    /// v2  += v1 *mul1 
    inline __DEVICE_TAG__ static void add_mul(scalar_type mul1, const Vec &v1, 
                                              Vec &v2);
    /// v2  = v1*mul1 
    inline __DEVICE_TAG__ static void assign_mul(scalar_type mul1 , const Vec &v1,
                                                 Vec &v2);
    /// v3  = v1+v2
    inline __DEVICE_TAG__ static void assign_sum(const Vec &v1 , const Vec &v2, 
                                                 Vec &v3);
    /// returns v1+v2
    inline __DEVICE_TAG__ static Vec sum(const Vec &v1 , const Vec &v2);
    /// returns v1-v2
    inline __DEVICE_TAG__ static Vec diff(const Vec &v1 , const Vec &v2);
    /// TODO add functions that returns result by value (like Vec sum(Vec,Vec))

    inline __DEVICE_TAG__ static scalar_type scalar_prod(const Vec &v1 , const Vec &v2);
    /// Only for dimension 3
    inline __DEVICE_TAG__ static Vec vector_prod(const Vec &v1 , const Vec &v2);
    inline __DEVICE_TAG__ static void assign_vector_prod(const Vec &v1 , const Vec &v2, Vec &v3);

    /// TODO add add_lin_comb and assign_lin_comb

    static_assert(!std::is_same<Vec,Vec>::value,
                  "static_vec_traits: not specialized");
};

template<class T,int Dim>
struct static_vec_traits<scfd::static_vec::vec<T,Dim>>
{
private:
    template<class X>using help_t = T;

public:
    static constexpr int dim = Dim;
    typedef scfd::static_vec::vec<T,Dim> vec_type;
    typedef T scalar_type;

    template<int J>
    inline __DEVICE_TAG__ static scalar_type &comp(vec_type &v1)
    {
        return v1.template get<J>();
    }
    template<int J>
    inline __DEVICE_TAG__ static const scalar_type &comp(const vec_type &v1)
    {
        return v1.template get<J>();
    }

    inline __DEVICE_TAG__ static scalar_type &x(vec_type &v1)
    {
        static_assert(dim > 0, "static_vec_traits::x: dim == 0");
        return v1.template get<0>();
    }
    inline __DEVICE_TAG__ static scalar_type &y(vec_type &v1)
    {
        static_assert(dim > 1, "static_vec_traits::y: dim <= 1");
        return v1.template get<1>();
    }
    inline __DEVICE_TAG__ static scalar_type &z(vec_type &v1)
    {
        static_assert(dim > 2, "static_vec_traits::z: dim <= 2");
        return v1.template get<2>();
    }
    inline __DEVICE_TAG__ static const scalar_type &x(const vec_type &v1)
    {
        static_assert(dim > 0, "static_vec_traits::x: dim == 0");
        return v1.template get<0>();
    }
    inline __DEVICE_TAG__ static const scalar_type &y(const vec_type &v1)
    {
        static_assert(dim > 1, "static_vec_traits::y: dim <= 1");
        return v1.template get<1>();
    }
    inline __DEVICE_TAG__ static const scalar_type &z(const vec_type &v1)
    {
        static_assert(dim > 2, "static_vec_traits::z: dim <= 2");
        return v1.template get<2>();
    }

    inline __DEVICE_TAG__ static vec_type zero()
    {
        return vec_type::make_zero();
    }
    template<typename... Args,
             class = typename std::enable_if<sizeof...(Args) == Dim>::type,
             class = typename std::enable_if<
                                  detail::check_all_are_true< 
                                      std::is_convertible<Args,help_t<Args> >::value... 
                                  >::value
                              >::type>
    inline __DEVICE_TAG__ static vec_type make(const Args&... args)
    {
        return vec_type(args...);
    }

    /// v2  = v1
    inline __DEVICE_TAG__ static void assign(const vec_type &v1, vec_type &v2)
    {
        v2 = v1;
    }
    /// v2  += v1
    inline __DEVICE_TAG__ static void add(const vec_type &v1, vec_type &v2)
    {
        v2 += v1;
    }
    /// v2  += v1*mul1 
    inline __DEVICE_TAG__ static void add_mul(scalar_type mul1, const vec_type &v1, 
                                              vec_type &v2)
    {
        v2  += v1*mul1;
    }
    /// v2  = v1*mul1 
    inline __DEVICE_TAG__ static void assign_mul(scalar_type mul1 , const vec_type &v1,
                                                 vec_type &v2)
    {
        v2  = v1*mul1;
    }
    /// v3  = v1+v2
    inline __DEVICE_TAG__ static void assign_sum(const vec_type &v1 , const vec_type &v2, 
                                                 vec_type &v3)
    {
        v3  = v1+v2;
    }
    /// returns v1+v2
    inline __DEVICE_TAG__ static vec_type sum(const vec_type &v1 , const vec_type &v2)
    {
        return v1+v2;
    }
    /// returns v1-v2
    inline __DEVICE_TAG__ static vec_type diff(const vec_type &v1 , const vec_type &v2)
    {
        return v1-v2;
    }

    inline __DEVICE_TAG__ static scalar_type scalar_prod(const vec_type &v1 , const vec_type &v2)
    {
        return static_vec::scalar_prod(v1,v2);
    }
    inline __DEVICE_TAG__ static vec_type vector_prod(const vec_type &v1 , const vec_type &v2)
    {
        static_assert(dim == 3, "static_vec_traits::vector_prod: only dim == 3 is supported");
        return static_vec::vector_prod(v1,v2);
    }
    inline __DEVICE_TAG__ static void assign_vector_prod(const vec_type &v1 , const vec_type &v2, 
                                                         vec_type &v3)
    {
        v3 = vector_prod(v1,v2);
    }

};

}
}

#endif
