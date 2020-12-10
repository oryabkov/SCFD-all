
#include <scfd/static_vec/vec.h>
#include <scfd/geometry/intersect_triangles.h>

#include "gtest/gtest.h"

namespace scfd
{
namespace geometry
{

using vec_t = scfd::static_vec::vec<float,3>;

class IntersectTrianglesTest : public ::testing::Test 
{
protected:
    bool check
    (
        vec_t t0_v0, vec_t t0_v1, vec_t t0_v2,
        vec_t t1_v0, vec_t t1_v1, vec_t t1_v2
    )
    {
        return intersect_triangles<float, vec_t>(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2);
    }
};

TEST_F(IntersectTrianglesTest, IntersectionByFullSegment) 
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(0.5f, 0.0f, 0.0f); 
    vec_t t1_v1(0.0f, 1.5f, 0.0f);
    vec_t t1_v2(0.0f, 0.0f, 1.5f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionByTriangleFace) 
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f, 0.f, 0.f);
    vec_t t1_v1(0.f, 1.f, 0.f);
    vec_t t1_v2(0.f, 0.f, 0.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionByTriangleFaceCoPlanar) 
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 0.f);

    vec_t t1_v0(1.f, 0.f, 0.f);
    vec_t t1_v1(0.f, 1.f, 0.f);
    vec_t t1_v2(1.f, 1.f, 0.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionByPartialSegment) 
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(0.7f, 0.0f, 0.0f);
    vec_t t1_v1(1.0f, 0.7f, 0.0f);
    vec_t t1_v2(1.0f, 0.7f, 0.2f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionCoincide) 
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f, 0.f, 0.f);
    vec_t t1_v1(0.f, 1.f, 0.f);
    vec_t t1_v2(0.f, 0.f, 1.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionBySide)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f, 0.f,  0.f);
    vec_t t1_v1(0.f, 1.f,  0.f);
    vec_t t1_v2(0.f, 0.f, -1.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionByPointInOrthogonalPlanes)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 0.f);

    vec_t t1_v0(0.f, 0.f,  0.f);
    vec_t t1_v1(-1.f, 0.f,  1.f);
    vec_t t1_v2(1.f, 0.f, 1.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}



TEST_F(IntersectTrianglesTest, IntersecitonByPoint)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f,  0.f,  0.f);
    vec_t t1_v1(0.f, -3.f,  0.f);
    vec_t t1_v2(0.f,  0.f, -3.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionInOneVerticeAndCoplanar)
{
    vec_t t0_v0(1.0f, 0.0f, 1.0f);
    vec_t t0_v1(0.0f, 1.0f, 1.0f);
    vec_t t0_v2(0.5f, 0.0f, 1.0f);

    vec_t t1_v0( -1.0f, 0.0f, 1.0f);
    vec_t t1_v1(  0.0f,-1.0f, 1.0f);
    vec_t t1_v2(  0.5f, 0.0f, 1.0f);    

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionVertice2FaceAndCoplanar)
{
    vec_t t0_v0(-1.0f, 0.0f, 1.0f);
    vec_t t0_v1( 1.0f, 0.0f, 1.0f);
    vec_t t0_v2( 0.0f, -1.0f, 1.0f);

    vec_t t1_v0( -1.0f, 1.0f, 1.0f);
    vec_t t1_v1(  1.0f, 1.0f, 1.0f);
    vec_t t1_v2(  0.0f, 0.0f, 1.0f);    

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, IntersectionBadAspectRatio)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(0.1f, 0.01f, 0.001f);
    vec_t t1_v1(100000000.f, 100000000.f, 10000000.f);
    vec_t t1_v2(100000000.f, -1000000000.f, -100000000.f);

    ASSERT_TRUE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, NoIntersectionVertice2FaceAndCoplanar)
{
    vec_t t0_v0(-1.0f, 0.0f, 1.0f);
    vec_t t0_v1( 1.0f, 0.0f, 1.0f);
    vec_t t0_v2( 0.0f, -1.0f, 1.0f);

    vec_t t1_v0( -1.0f, 1.0f, 1.0f);
    vec_t t1_v1(  1.0f, 1.0f, 1.0f);
    vec_t t1_v2(  0.0f, 1.0e-6f, 1.0f);    

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, NoIntersectionNearPointInOrthogonalPlanes)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 0.f);

    vec_t t1_v0(0.f, 0.f,  1.0e-6f);
    vec_t t1_v1(-1.f, 0.f,  1.f);
    vec_t t1_v2(1.f, 0.f, 1.f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, NoIntersectionQuasi2D)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 0.f);

    vec_t t1_v0(-1.f, 0.f, 0.f);
    vec_t t1_v1(-0.5f, 0.0f, 0.f);
    vec_t t1_v2(0.f, -1.f, 0.f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}


TEST_F(IntersectTrianglesTest, NoIntersectionNoNearIntersecitonByPoint)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f+0.000001f, 0.f, 0.f);
    vec_t t1_v1(0.f, -3.f, 0.f);
    vec_t t1_v2(0.f, 0.f, -3.f); 

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, NoIntersectionCoplanar)
{
    vec_t t0_v0(1.f, 0.f, 0.f);
    vec_t t0_v1(0.f, 1.f, 0.f);
    vec_t t0_v2(0.f, 0.f, 1.f);

    vec_t t1_v0(1.f+0.1f, 0.f, 0.f);
    vec_t t1_v1(0.f, 1.f+0.1f, 0.f);
    vec_t t1_v2(0.f, 0.f, 1.f+0.1f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, NoIntersectionCoplanar2) 
{
    /// Coplanar 2
    vec_t t0_v0(1.0f, 0.0f, 1.0f);
    vec_t t0_v1(0.0f, 1.0f, 1.0f);
    vec_t t0_v2(0.5f, 0.0f, 1.0f);

    vec_t t1_v0(-1.0f, 0.0f, 1.0f);
    vec_t t1_v1( 0.0f,-1.0f, 1.0f);
    vec_t t1_v2(-0.5f, 0.0f, 1.0f); 

    //random
    // vec_t t1_v0(1+0.1, 4, 2);
    // vec_t t1_v1(1, 1+0.1, -1);
    // vec_t t1_v2(2, 1, 1+0.1);   

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, ABAGYNoIntersectionCoplanar1) 
{
    vec_t t0_v0 = vec_t(4.509881019592285e+00f,8.474619388580322e-01f,-1.345000028610229e+00f);
    vec_t t0_v1 = vec_t(4.849546432495117e+00f,6.369206905364990e-01f,-1.345000028610229e+00f);
    vec_t t0_v2 = vec_t(4.849880695343018e+00f,8.469204306602478e-01f,-1.345000028610229e+00f);
    vec_t t1_v0 = vec_t(4.985392570495605e+00f,6.824411153793335e-01f,-1.345000028610229e+00f);
    vec_t t1_v1 = vec_t(5.025794029235840e+00f,7.118983268737793e-01f,-1.345000028610229e+00f);
    vec_t t1_v2 = vec_t(4.885238170623779e+00f,8.198057413101196e-01f,-1.345000028610229e+00f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));

    t0_v0 = vec_t(4.509881019592285e+00f,-1.345000028610229e+00f,8.474619388580322e-01f);
    t0_v1 = vec_t(4.849546432495117e+00f,-1.345000028610229e+00f,6.369206905364990e-01f);
    t0_v2 = vec_t(4.849880695343018e+00f,-1.345000028610229e+00f,8.469204306602478e-01f);
    t1_v0 = vec_t(4.985392570495605e+00f,-1.345000028610229e+00f,6.824411153793335e-01f);
    t1_v1 = vec_t(5.025794029235840e+00f,-1.345000028610229e+00f,7.118983268737793e-01f);
    t1_v2 = vec_t(4.885238170623779e+00f,-1.345000028610229e+00f,8.198057413101196e-01f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));

    t0_v0 = vec_t(-1.345000028610229e+00f,4.509881019592285e+00f,8.474619388580322e-01f);
    t0_v1 = vec_t(-1.345000028610229e+00f,4.849546432495117e+00f,6.369206905364990e-01f);
    t0_v2 = vec_t(-1.345000028610229e+00f,4.849880695343018e+00f,8.469204306602478e-01f);
    t1_v0 = vec_t(-1.345000028610229e+00f,4.985392570495605e+00f,6.824411153793335e-01f);
    t1_v1 = vec_t(-1.345000028610229e+00f,5.025794029235840e+00f,7.118983268737793e-01f);
    t1_v2 = vec_t(-1.345000028610229e+00f,4.885238170623779e+00f,8.198057413101196e-01f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

TEST_F(IntersectTrianglesTest, ABAGYNoIntersectionCoplanar2) 
{
    vec_t t0_v0 = vec_t(4.509881019592285e+00f,8.474619388580322e-01f,-1.345000028610229e+00f);
    vec_t t0_v1 = vec_t(4.849546432495117e+00f,6.369206905364990e-01f,-1.345000028610229e+00f);
    vec_t t0_v2 = vec_t(4.849880695343018e+00f,8.469204306602478e-01f,-1.345000028610229e+00f);
    vec_t t1_v0 = vec_t(4.885238170623779e+00f,8.198057413101196e-01f,-1.345000028610229e+00f);
    vec_t t1_v1 = vec_t(5.025794029235840e+00f,7.118983268737793e-01f,-1.345000028610229e+00f);
    vec_t t1_v2 = vec_t(4.925639629364014e+00f,8.492629528045654e-01f,-1.345000028610229e+00f);

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));

    t0_v0 = vec_t(-1.345000028610229e+00f,4.509881019592285e+00f,8.474619388580322e-01f);
    t0_v1 = vec_t(-1.345000028610229e+00f,4.849546432495117e+00f,6.369206905364990e-01f);
    t0_v2 = vec_t(-1.345000028610229e+00f,4.849880695343018e+00f,8.469204306602478e-01f);
    t1_v0 = vec_t(-1.345000028610229e+00f,4.885238170623779e+00f,8.198057413101196e-01f);
    t1_v1 = vec_t(-1.345000028610229e+00f,5.025794029235840e+00f,7.118983268737793e-01f);
    t1_v2 = vec_t(-1.345000028610229e+00f,4.925639629364014e+00f,8.492629528045654e-01f); 

    ASSERT_FALSE(check(t0_v0,t0_v1,t0_v2,t1_v0,t1_v1,t1_v2));
}

}
}