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
#include <vector>
#include <list>
#include <scfd/static_vec/vec.h>

#include "gtest/gtest.h"

using namespace scfd::static_vec;

TEST(StaticVecTest, InitFloatByValues) 
{
    vec<float,3>    v1(0.f,1.f,2);

    ASSERT_EQ(v1[0], 0.f);
    ASSERT_EQ(v1[1], 1.f);
    ASSERT_EQ(v1[2], 2.f);

    ASSERT_EQ(v1.get<0>(), 0.f);
    ASSERT_EQ(v1.get<1>(), 1.f);
    ASSERT_EQ(v1.get<2>(), 2.f);
}

TEST(StaticVecTest, InitIntByValues) 
{
    vec<int,3>          v2(0.5f,1.f,1);

    ASSERT_EQ(v2[0], 0);
    ASSERT_EQ(v2[1], 1);
    ASSERT_EQ(v2[2], 1);

    ASSERT_EQ(v2.get<0>(), 0);
    ASSERT_EQ(v2.get<1>(), 1);
    ASSERT_EQ(v2.get<2>(), 1);
}

TEST(StaticVecTest, InitIntBySTDVector) 
{
    std::vector<int>    vector_int = {1,2,3};
    vec<int,3>          v3(vector_int);     //calls tempalte 'Vec' constructor

    ASSERT_EQ(v3[0], 1);
    ASSERT_EQ(v3[1], 2);
    ASSERT_EQ(v3[2], 3);

    ASSERT_EQ(v3.get<0>(), 1);
    ASSERT_EQ(v3.get<1>(), 2);
    ASSERT_EQ(v3.get<2>(), 3);
}

TEST(StaticVecTest, InitIntByInitializerList) 
{
    vec<int,3>          v3 = {1,2,3.};

    ASSERT_EQ(v3[0], 1);
    ASSERT_EQ(v3[1], 2);
    ASSERT_EQ(v3[2], 3);

    ASSERT_EQ(v3.get<0>(), 1);
    ASSERT_EQ(v3.get<1>(), 2);
    ASSERT_EQ(v3.get<2>(), 3);
}

TEST(StaticVecTest, CopyConstructor) 
{
    vec<int,3>          v0(1,2,3);     //calls tempalte 'Vec' constructor
    vec<int,3>          v6(v0);        //calls normal copy constructor

    ASSERT_EQ(v6[0], 1);
    ASSERT_EQ(v6[1], 2);
    ASSERT_EQ(v6[2], 3);

    ASSERT_EQ(v6.get<0>(), 1);
    ASSERT_EQ(v6.get<1>(), 2);
    ASSERT_EQ(v6.get<2>(), 3);
}

TEST(StaticVecTest, SizeOfValue)
{
    ASSERT_EQ(sizeof(vec<float,3>), sizeof(float)*3);
    ASSERT_EQ(sizeof(vec<int,3>), sizeof(int)*3);
}

/*TEST(StaticVecTest, NoInitFromSTDList) 
{
    std::list<int>      list_int = {1,2,3};
    vec<int,3>          v4(list_int);     //not working - no operator[]
}*/

/*TEST(StaticVecTest, InitWithWrongDimension) 
{
    vec<int,3>          v5(0.5f,1.f);     //not working - number of argument
}*/

TEST(StaticVecTest, ScalarMul)
{
    vec<int,3>          v0(1,2,3),v1,v2;

    v1 = v0*2;
    ASSERT_EQ(v1[0], 2);
    ASSERT_EQ(v1[1], 4);
    ASSERT_EQ(v1[2], 6);
    v2 = 2*v0;
    ASSERT_EQ(v2[0], 2);
    ASSERT_EQ(v2[1], 4);
    ASSERT_EQ(v2[2], 6);
}
