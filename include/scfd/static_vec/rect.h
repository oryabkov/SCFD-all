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
    explicit __DEVICE_TAG__ rect(const vec_type &_i2) : i1(vec_type::make_zero()), i2(_i2) { }
    /// ISSUE implicit ok?
    template<class T2>
    __DEVICE_TAG__ rect(const rect<T2,Dim> &r) : i1(r.i1), i2(r.i2) { }

    __DEVICE_TAG__ bool is_own(const vec_type &p)const
    {
        for (int j = 0;j < Dim;++j)
            if (!((i1[j]<=p[j])&&(p[j]<i2[j]))) return false;
        return true;
    }
    __DEVICE_TAG__ vec<T,Dim>  calc_size()const { return i2-i1; }
    __DEVICE_TAG__ T           calc_size(int j)const { return i2[j]-i1[j]; }
    //TODO what about inversed rects where for some i2[j] < i1[j]? Think we should return zero for such case.
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
    /**
     * Creates rect which range contains all points from (-half_size,-half_size,...) 
     * up to (half_size,half_size,...) - borders included.
     * Use together with range for loops to walkthrough over all {-half_size,..,0,..,half_size} combinations
     * of indices. For example:
     * for (auto i : rect<int,2>::make_symm_square_range())
     * {
     *     ...
     * }
     * will iterate over (2*1+1)^2 = 9 corresponding indices.
     **/
    static __DEVICE_TAG__ rect make_symm_square_range(T half_size = T(1))
    {
        rect res;
        for (int j = 0;j < Dim;++j)
        {
            res.i1[j] = -half_size;
            res.i2[j] = half_size+1;
        }
        return res;
    }
    /**
     * Creates rect which range contains all points from (0,0,...) 
     * up to (half_size,half_size,...) - borders included.
     * Use together with range for loops to walkthrough over all {0,..,half_size} combinations
     * of indices. For example:
     * for (auto i : rect<int,3>::make_square_range(3))
     * {
     *     ...
     * }
     * will iterate over (3+1)^3 = 64 corresponding indices.
     **/
    static __DEVICE_TAG__ rect make_square_range(T size = T(1))
    {
        rect res;
        for (int j = 0;j < Dim;++j)
        {
            res.i1[j] = 0;
            res.i2[j] = size+1;
        }
        return res;
    }
    /// TODO better to rename it into intersection (intersect is a verb)
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
    /// Returns uniformly in all directions enlarged rect
    /// TODO what about initially empty rect?
    __DEVICE_TAG__ rect        enlarged(const T &pad_size)const
    {
        rect res = *this;
        for (int j = 0;j < Dim;++j)
        {
            res.i1[j] -= pad_size;
            res.i2[j] += pad_size;
        }
        return res;
    }
    __DEVICE_TAG__ rect        enlarged_one_dim(int dir, const T &pad_size)const
    {
        rect res = *this;
        res.i1[dir] -= pad_size;
        res.i2[dir] += pad_size;
        return res;
    }
    /// Returns padding rect in the neibourhood of the original one according to the following rules:
    /// For all dimensions j: padding_size[j]==0 dimensions leaved without changes.
    /// For all dimensions j: padding_size[j]<0 dimensions left (lower border) padding with absolute size -padding_size[j] is taken.
    /// For all dimensions j: padding_size[j]>0 dimensions right (upper border) padding with absolute size padding_size[j] is taken.
    /// TODO what about initially empty rect?
    __DEVICE_TAG__ rect        padding_rect(const vec<T, Dim> &padding_size)const
    {
        rect res = *this;
        for (int j = 0;j < Dim;++j)
        {
            if (padding_size[j] < 0) 
            {
                res.i2[j] = res.i1[j];
                res.i1[j] += padding_size[j];
            }
            else if (padding_size[j] > 0) 
            {
                res.i1[j] = res.i2[j];
                res.i2[j] += padding_size[j];
            }
        }
        return res;
    }
    __DEVICE_TAG__ rect        shifted(const vec<T, Dim> &idx_shift)const
    {
        rect res = *this;
        res.i1 += idx_shift;
        res.i2 += idx_shift;
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

    //this pair is more for for-style bypass (auto i = r.range_start();r.is_own(i);r.range_step(i))
    __DEVICE_TAG__ vec<T, Dim>    range_start()const
    {
        return i1;
    }
    __DEVICE_TAG__ void           range_step(vec<T, Dim> &idx)const
    {
        for (int j = Dim-1;j >= 0;--j) {
            ++(idx[j]);
            if (idx[j] < i2[j]) return;
            if (j == 0) return;  //we are over rect, so we leave idx[0] to be out of range which can be checked by is_own(idx)
            idx[j] = i1[j];
        }
        //we are never not here
    }

    //WARNING this version is for backward compability only and MUST NOT be used; use range_start/range_step instead
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

    class iterator 
    {
    public:
        __DEVICE_TAG__ iterator(rect r, vec_type idx): r_(r), idx_(idx) {}
        __DEVICE_TAG__ iterator operator++() 
        { 
            r_.range_step(idx_); 
            return *this; 
        }
        __DEVICE_TAG__ bool operator!=(const iterator & other) const 
        { 
            //TODO
            //if (r_ != other.r_) return false;  /// TODO think its more like exception with logic
            return idx_ != other.idx_;  
        }
        /// TODO by value ok?
        __DEVICE_TAG__ vec_type operator*() const { return idx_; }
    private:
        rect r_; 
        vec_type idx_;
    };

    __DEVICE_TAG__ iterator begin() const 
    { 
        return iterator(*this, i1); 
    }
    __DEVICE_TAG__ iterator end() const 
    { 
        /// according to logic of _bypass_step
        vec_type end_idx = i1;
        end_idx[0] = i2[0];
        return iterator(*this, end_idx); 
    }

    /**
     * use this to perform indexed loops over the rect range (for example to use omp):
     * for (T i = 0;i < r.calc_area();++i)
     * {
     *      auto idx = lin_idx_2_nd_idx(i);
     *      ...
     * }
     * NOTE indexing is only c-style for now i.e. last index fast
     **/
    __DEVICE_TAG__ vec_type lin_idx_2_nd_idx(T lin_idx)const
    {
        vec_type res;
        //TODO this infact a precondition - should raise error?
        if (is_empty()) return vec_type::make_zero();
        for (int j = Dim-1;j >= 0;--j)
        {
            T sz_j = calc_size(j);
            res[j] = i1[j] + lin_idx%sz_j;
            lin_idx /= sz_j;
        }
        return res;
    }
};

}

}

#endif
