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


#include <scfd/arrays/detail/parameter_indexer.h>

#include "gtest/gtest.h"

TEST(DetailParameterIndexerTest, GetMixedTests) 
{
    /// NOTE template argumnets after the first one are ignored - just therir number is important
    access_index<0, 1,1,1>  test1;
    access_index<1, 2,2,1>  test2;
    access_index<2, 1,3,3>  test3;

    ASSERT_EQ((test1.get(1,2,6,10)), 1);
    ASSERT_EQ((test2.get(1,2,6,11)), 2);
    ASSERT_EQ((test3.get(1,2,6,12)), 6);
}
