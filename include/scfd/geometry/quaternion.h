#ifndef __SCFD_GEOMETRY_QUATERNION_H__
#define __SCFD_GEOMETRY_QUATERNION_H__

#include <cmath>
#include <scfd/utils/scalar_traits.h>
#include "static_vec_traits.h"

namespace scfd
{
namespace geometry
{

/// Vec must satisfy StaticVec concept
template<class Vec>
struct quaternion
{
private:
    typedef static_vec_traits<Vec>                           vt;
    typedef utils::scalar_traits<typename vt::scalar_type>   st;

    static_assert(vt::dim == 3, 
                  "quaternion: only 3 dimensional Vec is supported");

public:
    typedef Vec                                vec_type;
    typedef typename vt::scalar_type           scalar_type;

public:
    scalar_type     s;
    vec_type        v;

public:
    __DEVICE_TAG__ quaternion() = default;
    __DEVICE_TAG__ quaternion(scalar_type s, scalar_type x, scalar_type y, scalar_type z)
    {
        this->s = s;
        v = vt::make(x,y,z);
    }
    __DEVICE_TAG__ ~quaternion() = default;

    __DEVICE_TAG__ static quaternion make_rotation(scalar_type angle, scalar_type unit_axis[3])
    {
        quaternion  res;
        res.s = std::cos(angle/2.0f);
        scalar_type sin2 = std::sin(angle/2.0f);
        res.v.x = sin2 * unit_axis[0];
        res.v.y = sin2 * unit_axis[1];
        res.v.z = sin2 * unit_axis[2];
        return res;
    }
    __DEVICE_TAG__ static quaternion make_rotation_x(scalar_type angle)
    {
        quaternion  res;
        res.s = std::cos(angle/2.0f);
        scalar_type sin2 = std::sin(angle/2.0f);
        res.v.x = sin2;
        res.v.y = 0.f;
        res.v.z = 0.f;
        return res;
    }
    __DEVICE_TAG__ static quaternion make_rotation_y(scalar_type angle)
    {
        quaternion  res;
        res.s = std::cos(angle/2.0f);
        scalar_type sin2 = std::sin(angle/2.0f);
        res.v.x = 0.f;
        res.v.y = sin2;
        res.v.z = 0.f;
        return res;
    }
    __DEVICE_TAG__ static quaternion make_rotation_z(scalar_type angle)
    {
        quaternion  res;
        res.s = std::cos(angle/2.0f);
        scalar_type sin2 = std::sin(angle/2.0f);
        res.v.x = 0.f;
        res.v.y = 0.f;
        res.v.z = sin2;
        return res;
    }

    __DEVICE_TAG__ quaternion operator*(const quaternion &q2) const 
    {
        quaternion res(
            s * q2.s - vt::x(v) * vt::x(q2.v) - vt::y(v)    * vt::y(q2.v) - vt::z(v) * vt::z(q2.v),
            s * vt::x(q2.v) + q2.s * vt::x(v) + vt::y(v)    * vt::z(q2.v) - vt::y(q2.v) * vt::z(v),
            s * vt::y(q2.v) + q2.s * vt::y(v) + vt::x(q2.v) * vt::z(v)    - vt::x(v)    * vt::z(q2.v),
            s * vt::z(q2.v) + q2.s * vt::z(v) + vt::x(v)    * vt::y(q2.v) - vt::x(q2.v) * vt::y(v));

        return res;
    }

    /// Inplace invert
    __DEVICE_TAG__ void invert()
    {
        v = -v;
    }

    __DEVICE_TAG__ void rotate_vec(const vec_type& v_, vec_type& v_prime)const
    {
        vt::assign_mul(scalar_type(2) * vt::scalar_prod(v, v_), v, v_prime);
        vt::add_mul(s*s - vt::scalar_prod(v, v), v_, v_prime);
        vt::add_mul(scalar_type(2) * s, vt::vector_prod(v, v_), v_prime);
    }

};

}
}

#endif