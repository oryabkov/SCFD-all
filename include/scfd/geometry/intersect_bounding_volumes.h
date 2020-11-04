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
 
#ifndef __SCFD_GEOMETRY_INTERSECT_BOUNDING_VOLUMES_H_
#define __SCFD_GEOMETRY_INTERSECT_BOUNDING_VOLUMES_H_

#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/static_vec/vec.h>
#include "obb.h"

/**
* TODO: 
* Would be cool to template it on vec type (need vec traits)
*/

#define SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON 0.000001f

namespace scfd
{
namespace geometry
{

/*template <class BV>
__DEVICE_TAG__ bool intersect_bounding_volumes(const BV &node1, const BV &node2)
{
    return true;
}*/

/// 
template <class T>
__DEVICE_TAG__ bool intersect_bounding_volumes(const obb<T> &node1, const obb<T> &node2)
{
    typedef typename obb<T>::scalar_type            scalar_t;
    typedef scfd::utils::scalar_traits<scalar_t>    st;
    typedef typename obb<T>::vec_type               vec_t;

    using static_vec::scalar_prod;

    //translation, in parent frame
    vec_t v = node1.center-node2.center;
    
    //translation, in A's frame
    vec_t tt(scalar_prod(v, node1.axis1), scalar_prod(v, node1.axis2), scalar_prod(v, node1.axis3));
    
    
    //calculate rotation matrix (B's basis with respect to A', R1)
    vec_t R1;
    R1[0] = scalar_prod(node1.axis1, node2.axis1);
    R1[1] = scalar_prod(node1.axis1, node2.axis2);
    R1[2] = scalar_prod(node1.axis1, node2.axis3);

    /*printf("intersect_bounding_volumes: R1[0] = %f\n", R1[0]);
    printf("intersect_bounding_volumes: R1[1] = %f\n", R1[1]);
    printf("intersect_bounding_volumes: R1[2] = %f\n", R1[2]);*/
    
    /*
    ALGORITHM: Use the separating axis test for all 15 potential
    separating axes. If a separating axis could not be found, the two
    boxes overlap.
    */
    
    // Axes: A's basis vectors
    {
        scalar_t rb;
        rb = node2.extents[0] * st::abs(R1[0]) + node2.extents[1] * st::abs(R1[1]) + node2.extents[2] * st::abs(R1[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1] + node2.extents[2]);
        if(st::abs(tt[0]) > (node1.extents[0] + rb))
            return false;
    }
    
    //calculate rotation matrix (B's basis with respect to A', R2)
    vec_t R2;
    R2[0] = scalar_prod(node1.axis2, node2.axis1);
    R2[1] = scalar_prod(node1.axis2, node2.axis2);
    R2[2] = scalar_prod(node1.axis2, node2.axis3);
    
    {
        scalar_t rb;
        rb = node2.extents[0] * st::abs(R2[0]) + node2.extents[1] * st::abs(R2[1]) + node2.extents[2] * st::abs(R2[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1] + node2.extents[2]);
        if(st::abs(tt[1]) > (node1.extents[1] + rb))
            return false;
    }
    
    //calculate rotation matrix (B's basis with respect to A', R3)
    vec_t R3;
    R3[0] = scalar_prod(node1.axis3, node2.axis1);
    R3[1] = scalar_prod(node1.axis3, node2.axis2);
    R3[2] = scalar_prod(node1.axis3, node2.axis3);
    
    {
        scalar_t rb;
        rb = node2.extents[0] * st::abs(R3[0]) + node2.extents[1] * st::abs(R3[1]) + node2.extents[2] * st::abs(R3[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1] + node2.extents[2]);;
        if(st::abs(tt[2]) > (node1.extents[2] + rb))
            return false;
    }
    
    // Axes: B's basis vectors
    {
        scalar_t rb, t;
        rb = node1.extents[0] * st::abs(R1[0]) + node1.extents[1] * st::abs(R2[0]) + node1.extents[2] * st::abs(R3[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1] + node1.extents[2]);;
        t = st::abs(tt[0] * R1[0] + tt[1] * R2[0] + tt[2] * R3[0]);
        if(t > (node2.extents[0] + rb))
            return false;
            
        rb = node1.extents[0] * st::abs(R1[1]) + node1.extents[1] * st::abs(R2[1]) + node1.extents[2] * st::abs(R3[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1] + node1.extents[2]);;;
        t = st::abs(tt[0] * R1[1] + tt[1] * R2[1] + tt[2] * R3[1]);
        if(t > (node2.extents[1] + rb))
            return false;
            
        rb = node1.extents[0] * st::abs(R1[2]) + node1.extents[1] * st::abs(R2[2]) + node1.extents[2] * st::abs(R3[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1] + node1.extents[2]);;;
        t = st::abs(tt[0] * R1[2] + tt[1] * R2[2] + tt[2] * R3[2]);
        if(t > (node2.extents[2] + rb))
            return false;
    }
    
    // Axes: 9 cross products
    
    //L = A0 x B0
    {
        scalar_t ra, rb, t;
        ra = node1.extents[1] * st::abs(R3[0]) + node1.extents[2] * st::abs(R2[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[1] + node1.extents[2]);
        rb = node2.extents[1] * st::abs(R1[2]) + node2.extents[2] * st::abs(R1[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[1] + node2.extents[2]);
        t = st::abs(tt[2] * R2[0] - tt[1] * R3[0]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A0 x B1
    {
        scalar_t ra, rb, t;
        ra = node1.extents[1] * st::abs(R3[1]) + node1.extents[2] * st::abs(R2[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[1] + node1.extents[2]);
        rb = node2.extents[0] * st::abs(R1[2]) + node2.extents[2] * st::abs(R1[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[2]);
        t = st::abs(tt[2] * R2[1] - tt[1] * R3[1]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A0 x B2
    {
        scalar_t ra, rb, t;
        ra = node1.extents[1] * st::abs(R3[2]) + node1.extents[2] * st::abs(R2[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[1] + node1.extents[2]);
        rb = node2.extents[0] * st::abs(R1[1]) + node2.extents[1] * st::abs(R1[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1]);
        t = st::abs(tt[2] * R2[2] - tt[1] * R3[2]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A1 x B0
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R3[0]) + node1.extents[2] * st::abs(R1[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[2]);
        rb = node2.extents[1] * st::abs(R2[2]) + node2.extents[2] * st::abs(R2[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[1] + node2.extents[2]);
        t = st::abs(tt[0] * R3[0] - tt[2] * R1[0]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A1 x B1
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R3[1]) + node1.extents[2] * st::abs(R1[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[2]);
        rb = node2.extents[0] * st::abs(R2[2]) + node2.extents[2] * st::abs(R2[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[1] + node2.extents[2]);
        t = st::abs(tt[0] * R3[1] - tt[2] * R1[1]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A1 x B2
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R3[2]) + node1.extents[2] * st::abs(R1[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[2]);
        rb = node2.extents[0] * st::abs(R2[1]) + node2.extents[1] * st::abs(R2[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1]);
        t = st::abs(tt[0] * R3[2] - tt[2] * R1[2]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A2 x B0
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R2[0]) + node1.extents[1] * st::abs(R1[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1]);
        rb = node2.extents[1] * st::abs(R3[2]) + node2.extents[2] * st::abs(R3[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[1] + node2.extents[2]);
        t = st::abs(tt[1] * R1[0] - tt[0] * R2[0]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A2 x B1
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R2[1]) + node1.extents[1] * st::abs(R1[1]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1]);
        rb = node2.extents[0] * st::abs(R3[2]) + node2.extents[2] * st::abs(R3[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[2]);
        t = st::abs(tt[1] * R1[1] - tt[0] * R2[1]);
        
        if(t > ra + rb)
            return false;
    }
    
    //L = A2 x B2
    {
        scalar_t ra, rb, t;
        ra = node1.extents[0] * st::abs(R2[2]) + node1.extents[1] * st::abs(R1[2]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node1.extents[0] + node1.extents[1]);
        rb = node2.extents[0] * st::abs(R3[1]) + node2.extents[1] * st::abs(R3[0]) + SCFD_GEOMETRY_OBB_ROTATION_MATRIX_EPSILON * (node2.extents[0] + node2.extents[1]);
        t = st::abs(tt[1] * R1[2] - tt[0] * R2[2]);
        
        if(t > ra + rb)
            return false;
    }
    
    // no separating axis found:
    // the two boxes overlap
    
    return true;
}

}
}

#endif