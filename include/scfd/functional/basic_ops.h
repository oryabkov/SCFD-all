// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_FUNCTIONAL_BASIC_OPS_H__
#define __SCFD_FUNCTIONAL_BASIC_OPS_H__

#include <scfd/utils/device_tag.h>

namespace scfd
{
namespace functional
{

template <class T>
struct plus
{
    __DEVICE_TAG__ T operator()( const T &a, const T &b ) const
    {
        return a + b;
    }
};

template <class T>
struct minimum
{
    __DEVICE_TAG__ T operator()( const T &a, const T &b ) const
    {
        return b < a ? b : a;
    }
};

template <class T>
struct maximum
{
    __DEVICE_TAG__ T operator()( const T &a, const T &b ) const
    {
        return a < b ? b : a;
    }
};

template <class T>
struct equal_to
{
    __DEVICE_TAG__ bool operator()( const T &a, const T &b ) const
    {
        return a == b;
    }
};

template <class T>
struct less
{
    __DEVICE_TAG__ bool operator()( const T &a, const T &b ) const
    {
        return a < b;
    }
};

}
}

#endif
