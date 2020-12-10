
#include <scfd/static_vec/vec.h>
#include <scfd/geometry/static_vec_traits.h>

#include "gtest/gtest.h"

namespace scfd
{
namespace geometry
{

using real = float;
using vec3_t = static_vec::vec<real,3>;
using vt = static_vec_traits<vec3_t>;

class StaticVecTraitsTest : public ::testing::Test 
{
protected:
    StaticVecTraitsTest()
    {
        a = vt::make(1.f, 0.f, 5.f);
        b = vt::make(0.f,-2.f, 3.f);
    }

    vec3_t  a,b,c;
};

TEST_F(StaticVecTraitsTest, Access) 
{
    real    a_x = vt::x(a),
            a_y = vt::y(a),
            a_z = vt::z(a);

    EXPECT_EQ(a_x, 1.f);
    EXPECT_EQ(a_y, 0.f);
    EXPECT_EQ(a_z, 5.f);

    real    b_x = vt::comp<0>(b),
            b_y = vt::comp<1>(b),
            b_z = vt::comp<2>(b);

    EXPECT_EQ(b_x, 0.f);
    EXPECT_EQ(b_y,-2.f);
    EXPECT_EQ(b_z, 3.f);
}

TEST_F(StaticVecTraitsTest, SumDiffOps) 
{
    vt::assign(a,c);
    EXPECT_EQ(vt::x(c), 1.f);
    EXPECT_EQ(vt::y(c), 0.f);
    EXPECT_EQ(vt::z(c), 5.f);

    vt::add(a,c);
    EXPECT_EQ(vt::x(c), 2.f);
    EXPECT_EQ(vt::y(c), 0.f);
    EXPECT_EQ(vt::z(c), 10.f);

    vt::add_mul(2.f,b,c);
    EXPECT_EQ(vt::x(c), 2.f);
    EXPECT_EQ(vt::y(c),-4.f);
    EXPECT_EQ(vt::z(c),16.f);
}

}
}