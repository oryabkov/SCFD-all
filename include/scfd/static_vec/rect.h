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

#ifndef __SCFD_RECT_H__
#define __SCFD_RECT_H__

#include <scfd/utils/device_tag.h>
#include "vec.h"

namespace scfd
{
namespace static_vec 
{

//NOTE T is supposed to be int or other ordinal (not real number)
template<class T,int Dim>
struct rect
{
    using vec_type = vec<T,Dim>;

    vec_type  i1, i2;
    __DEVICE_TAG__ rect() { }
    __DEVICE_TAG__ rect(const vec_type &_i1, const vec_type &_i2) : i1(_i1), i2(_i2) { }

    __DEVICE_TAG__ bool is_own(const vec_type &p)const
    {
        for (int j = 0;j < Dim;++j)
            if (!((i1[j]<=p[j])&&(p[j]<i2[j]))) return false;
        return true;
    }
    __DEVICE_TAG__ vec<T,Dim>  calc_size()const { return i2-i1; }
    __DEVICE_TAG__ T           calc_area()const 
    { 
        T   res(1);
        for (int j = 0;j < Dim;++j)
            res *= (i2[j] - i1[j]);
        return res;
    }
    static __DEVICE_TAG__ rect make_empty()
    {
        return rect<T,Dim>(vec_type::make_zero(),vec_type::make_zero());
    }
    __DEVICE_TAG__ rect        intersect(const rect &r)
    {
        rect res;
        bool is_empty = false;
        for (int j = 0;j < Dim;++j)
        {
            //TODO correct min/max functions (need traits)
            res.i1[j] = max(i1[j],r.i1[j]);
            res.i2[j] = min(i2[j],r.i2[j]);
            if (res.i1[j] >= res.i2[j]) is_empty = true;
        }
        if (is_empty) res = make_empty();
        return res;
    }

    __DEVICE_TAG__ bool         is_empty()const
    {
        bool is_empty = false;
        for (int j = 0;j < Dim;++j)
        {
            if (i1[j] >= i2[j]) is_empty = true;
        }
        return is_empty;
    }

    __DEVICE_TAG__ bool         bypass_start(vec<T, Dim> &idx)const
    {
        if (is_empty()) return false;
        idx = i1;
        return true;
    }
    __DEVICE_TAG__ bool         bypass_step(vec<T, Dim> &idx)const
    {
        for (int j = Dim-1;j >= 0;--j) {
            ++(idx[j]);
            if (idx[j] < i2[j]) return true;
            idx[j] = i1[j];
        }
        return false;
    }

    //ISSUE what about names (_-prefix)??
    //this pair is more for for-style bypass (t_idx i = r.bypass_start();r.is_own(i);r.bypass_step(i))
    __DEVICE_TAG__ vec<T, Dim>    _bypass_start()const
    {
        return i1;
    }
    __DEVICE_TAG__ void                 _bypass_step(vec<T, Dim> &idx)const
    {
        for (int j = Dim-1;j >= 0;--j) {
            ++(idx[j]);
            if (idx[j] < i2[j]) return;
            if (j == 0) return;  //we are over rect, so we leave idx[0] to be out of range which can be checked by is_own(idx)
            idx[j] = i1[j];
        }
        //we are never not here
    }


};

}

}

#endif
