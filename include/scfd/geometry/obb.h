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
 
#ifndef __SCFD_GEOMETRY_OBB_H_
#define __SCFD_GEOMETRY_OBB_H_

#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include "quaternion.h"

namespace scfd
{
namespace geometry
{

/**
* Oriented bounding box
*/
template<class T>
class obb
{
public:
    typedef T                     scalar_type;
    typedef static_vec::vec<T,3>  vec_type;
    typedef quaternion<vec_type>  quaternion_type;
    
public:
    __DEVICE_TAG__ T get_size()const
    {
        return extents[0] * extents[0] + extents[1] * extents[1] + extents[2] * extents[2];
    }
    __DEVICE_TAG__ void rotate(const vec_type &p_0, 
                               const quaternion_type &q)
    {
        vec_type  v_res;
        q.rotate_vec(axis1, v_res);
        axis1 = v_res;
        q.rotate_vec(axis2, v_res);
        axis2 = v_res;
        q.rotate_vec(axis3, v_res);
        axis3 = v_res;
        q.rotate_vec(center, v_res);
        v_res += p_0;
        center = v_res;
    }
    
    /// center point of OBB
    vec_type center;
    /// major axes specifying the OBB
    vec_type axis1,
             axis2,
             axis3;
    /// extents to boundary from center point, along each axis
    vec_type extents;     
};

}
}

#endif
