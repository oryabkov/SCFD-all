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
 
#ifndef __SCFD_GEOMETRY_OBB_NODE_H_
#define __SCFD_GEOMETRY_OBB_NODE_H_

#include <scfd/utils/device_tag.h>
#include "obb.h"

/// TODO 
/// 1) delete right
/// 2) remake all this gProximity nonsence about 
///    2 and 5 shift - what's it all about
/// 3) what about ordinal type? mb, template?
/// 4) whay get_left_child and get_tri_ind return 
///    signed value? Is there any logic in this?

namespace scfd
{
namespace geometry
{

template<class T>
class obb_node
{
public:
    obb<T> bbox;                // bounding box for node
    unsigned int left,          // pointers to left/right children
                 right;
                     
    __DEVICE_TAG__ bool is_leaf()const
    {
        return (left & 3) == 3;
    }
    
    __DEVICE_TAG__ int get_left_child() const
    {
        return left >> 5;
    }
    __DEVICE_TAG__ int get_tri_ind() const
    {
        return left >> 2;
    }
};

}
}

#endif
