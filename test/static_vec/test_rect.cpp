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

#include <iostream>
#include <scfd/static_vec/rect.h>

#include "gtest/gtest.h"

using namespace scfd::static_vec;

typedef vec<int,3>    idx_t;
typedef rect<int,3>   rect_t;

TEST(StaticRectTest, CalcArea) 
{
    rect_t  r(idx_t(0,0,0), idx_t(1,1,1));

    ASSERT_EQ(r.calc_area(), 1);
}
