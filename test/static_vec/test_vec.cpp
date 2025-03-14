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

TEST(StaticVecTest, ScalarMulInt)
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
    /// Check inplace and reuse
    v1 = v0;
    v1 *= 3;
    ASSERT_EQ(v1[0], 3);
    ASSERT_EQ(v1[1], 6);
    ASSERT_EQ(v1[2], 9);

    v1 = v0*1.5;  ///converted to 1
    ASSERT_EQ(v1[0], 1);
    ASSERT_EQ(v1[1], 2);
    ASSERT_EQ(v1[2], 3);       
}

TEST(StaticVecTest, ScalarMulFloat)
{
    vec<float,3>          v0(1,2,3),v1,v2;

    v1 = v0*2.5f;
    ASSERT_EQ(v1[0], 2.5f);
    ASSERT_EQ(v1[1], 5.f);
    ASSERT_EQ(v1[2], 7.5f);
    v2 = 2.5f*v0;
    ASSERT_EQ(v2[0], 2.5f);
    ASSERT_EQ(v2[1], 5.f);
    ASSERT_EQ(v2[2], 7.5f);
    /// Check inplace and reuse
    v1 = v0;
    v1 *= 3.5f;
    ASSERT_EQ(v1[0], 3.5f);
    ASSERT_EQ(v1[1], 7.f);
    ASSERT_EQ(v1[2], 10.5f);
}

TEST(StaticVecTest, ScalarDivInt)
{
    vec<int,3>          v0(1,2,3),v1,v2;

    v1 = v0/2;
    ASSERT_EQ(v1[0], 0);
    ASSERT_EQ(v1[1], 1);
    ASSERT_EQ(v1[2], 1);

    v2 = v0/2.5;  ///converted to 2
    ASSERT_EQ(v2[0], 0);
    ASSERT_EQ(v2[1], 1);
    ASSERT_EQ(v2[2], 1); 

    /// Check inplace and reuse
    v1 = v0;
    v1 /= 3;
    ASSERT_EQ(v1[0], 0);
    ASSERT_EQ(v1[1], 0);
    ASSERT_EQ(v1[2], 1);
}

TEST(StaticVecTest, ScalarDivFloat)
{
    vec<float,3>          v0(1,2,3),v1,v2;

    v1 = v0/2;
    ASSERT_EQ(v1[0], 0.5f);
    ASSERT_EQ(v1[1], 1.f);
    ASSERT_EQ(v1[2], 1.5f);

    v2 = v0/0.5f;  
    ASSERT_EQ(v2[0], 2.f);
    ASSERT_EQ(v2[1], 4.f);
    ASSERT_EQ(v2[2], 6.f); 

    /// Check inplace and reuse
    v1 = v0;
    v1 /= 0.25f;
    ASSERT_EQ(v1[0], 4.f);
    ASSERT_EQ(v1[1], 8.f);
    ASSERT_EQ(v1[2], 12.f);
}

TEST(StaticVecTest, MakeOnesAndScalarDivMixed)
{
    vec<float,3>          vf = vec<float,3>::make_ones()/2;
    ASSERT_EQ(vf[0], 1.f/2.f);
    ASSERT_EQ(vf[1], 1.f/2.f);
    ASSERT_EQ(vf[2], 1.f/2.f); 
}

TEST(StaticVecTest, LinCombMixed)
{
    float h = 1.f/100.f;
    vec<float,3> hx = {h, 0, 0};
    vec<float,3> hy = {0, h, 0};
    vec<float,3> hz = {0, 0, h};

    int i = 1,j = 2,k = 3; 
    auto xyz = hx*i + hy*j + hz*k;
    ASSERT_EQ(xyz[0], 1.f/100.f);
    ASSERT_EQ(xyz[1], 2.f/100.f);
    ASSERT_EQ(xyz[2], 3.f/100.f);    

    auto xyz2 = i*hx + j*hy + k*hz;
    ASSERT_EQ(xyz2[0], 1.f/100.f);
    ASSERT_EQ(xyz2[1], 2.f/100.f);
    ASSERT_EQ(xyz2[2], 3.f/100.f); 
}
