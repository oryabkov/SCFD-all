#ifndef __SCFD_GEOMETRY_INTERSECT_TRIANGLES_H__
#define __SCFD_GEOMETRY_INTERSECT_TRIANGLES_H__

#include <cstdio>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include "static_vec_traits.h"

/// FROM:
/// https://github.com/kenny-evitt/three-js-triangle-triangle-collision-detection/blob/master/collision-tests.js

namespace scfd
{
namespace geometry
{

template<typename T>
__DEVICE_TAG__ bool are_projections_separated(const T &p0, const T &p1, const T &p2, const T &q0, const T &q1, const T &q2)
{
    
    T min_p = fminf(p0, fminf(p1, p2));
    T max_p = fmaxf(p0, fmaxf(p1, p2));
    T min_q = fminf(q0, fminf(q1, q2));
    T max_q = fmaxf(q0, fmaxf(q1, q2));

    return ((min_p > max_q) || (max_p < min_q));

}

template<typename T, class Vec>
__DEVICE_TAG__ bool are_projections_separated(const Vec &v0, const Vec &v1)
{
    typedef static_vec_traits<Vec> vt;

    T p0 = vt::x(v0);
    T p1 = vt::y(v0);
    T p2 = vt::z(v0);
    T q0 = vt::x(v1);
    T q1 = vt::y(v1);
    T q2 = vt::z(v1);

    return( are_projections_separated<T>(p0, p1, p2, q0, q1, q2) );

}

//main checker
template<typename T, class Vec>
__DEVICE_TAG__ bool intersect_triangles(const Vec &t0_v0, const Vec &t0_v1, const Vec &t0_v2, const Vec &t1_v0, const Vec &t1_v1,const Vec &t1_v2)
{
    typedef static_vec_traits<Vec> vt;

    const auto arePS = are_projections_separated<T,Vec>;


    Vec A0 = t0_v0;
    Vec A1 = t0_v1;
    Vec A2 = t0_v2;
    Vec E0 = vt::diff(A1,A0);
    Vec E1 = vt::diff(A2,A0);
    Vec E2 = vt::diff(E1,E0); //A0? Original was E1-E0
    Vec N = vt::vector_prod(E0,E1); 

    Vec B0 = t1_v0;
    Vec B1 = t1_v1;
    Vec B2 = t1_v2;
    Vec F0 = vt::diff(B1,B0);
    Vec F1 = vt::diff(B2,B0);
    Vec F2 = vt::diff(F1,F0); //B0? Original was F1-F0
    Vec M = vt::vector_prod(F0,F1);
    
    Vec D = vt::diff(B0,A0);


//  Check axis N
    T NdotD = vt::scalar_prod(N,D);
    Vec p0(T(0),T(0),T(0));
    Vec q0(NdotD, NdotD + vt::scalar_prod(N,F0), NdotD + vt::scalar_prod(N,F1) );
    if ( arePS(p0, q0) )
    {
        //printf("failed by: 1\n");
        return false;      
    }
//  Check axis M
    T MdotD = vt::scalar_prod(M,D);
    Vec p1(T(0), vt::scalar_prod(M,E0), vt::scalar_prod(M,E1) );
    Vec q1(MdotD, MdotD, MdotD);
    if ( arePS(p1, q1) )
    {
        //printf("failed by: 2\n");
        return false;    
    }

//Check axis Ej \times Fk where {j,k}=0,1,2

    Vec Eall[3] = {E0, E1, E2};
    Vec Fall[3] = {F0, F1, F2};

    for(int j=0; j<3; j++)
    {
        T flag_j = (j==0?T(-1):T(1));

        for(int k=0; k<3;k++)
        {
            T flag_k = (k==0?T(1):T(-1));

            Vec E = Eall[j];
            Vec F = Fall[k];
            T ExFdD = vt::scalar_prod(vt::vector_prod(E,F),D);
            Vec p2 = vt::make(T(0),T(0),flag_j*vt::scalar_prod(N,F));
            Vec q2 = vt::make(ExFdD, ExFdD, ExFdD + flag_k*vt::scalar_prod(M,E) );
            if ( arePS(p2, q2) )
            {
                //printf("failed by: flag_j = %f, flag_k = %f, j=%i, k=%i\n",flag_j, flag_k, j, k);
                return false;    
            }
        }
    }
    
//testing co-planar triangles! 
//Error in the original method. Other axis are needed to seperate triangles for the co-planar case.
// l = ||N||; E_j X F_k = lambda_{j,k} N, j,k=0,1; F_0 X F_1 = mu N

    // This part does the same as in line 75. WTF did i put it here??
    // q0 = vt::make(NdotD, NdotD + vt::scalar_prod(N,F0), NdotD + vt::scalar_prod(N,F1));
    // if ( arePS(p0, q0) )
    // {
    //     //printf("failed by: co-plane 1\n");
    //     return false;      
    // }

    T sqN = vt::scalar_prod(N,N);

    T lambda_00 = vt::scalar_prod(vt::vector_prod(E0, F0),N);
    T lambda_01 = vt::scalar_prod(vt::vector_prod(E0, F1),N);
    T lambda_11 = vt::scalar_prod(vt::vector_prod(E1, F1),N);
    T lambda_10 = vt::scalar_prod(vt::vector_prod(E1, F0),N);

    T mu = vt::scalar_prod(vt::vector_prod(F0, F1),N);

    //filling  (N X E_j, D) and (N X F_j, D), j = {0,1,2}
    T q_0[6];
    for(int j = 0; j<3; j++)
    {
        q_0[j] = vt::scalar_prod(vt::vector_prod(N, Eall[j]),D);
        q_0[j+3] = vt::scalar_prod(vt::vector_prod(N, Fall[j]),D);
    }

    p0 = vt::make(T(0), T(0), sqN);
    q0 = vt::make(q_0[0], q_0[0] + lambda_00, q_0[0] + lambda_01); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 2\n"); 
        return false;   
    }

    p0 = vt::make(T(0), -sqN, T(0));
    q0 = vt::make(q_0[1], q_0[1] + lambda_10, q_0[1] + lambda_11); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 3\n"); 
        return false;   
    }

    p0 = vt::make(T(0), -sqN, -sqN);
    q0 = vt::make(q_0[2], q_0[2] + (lambda_10-lambda_00), q_0[2] + (lambda_11-lambda_01)); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 4\n"); 
        return false;   
    }

    p0 = vt::make(T(0), -lambda_00, -lambda_10);
    q0 = vt::make(q_0[3], q_0[3], q_0[3] + mu); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 5\n"); 
        return false;   
    }

    p0 = vt::make(T(0), -lambda_01, -lambda_11);
    q0 = vt::make(q_0[4], q_0[4] - mu, q_0[4]); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 6\n"); 
        return false;   
    }    

    p0 = vt::make(T(0), (lambda_00-lambda_01), (lambda_10 - lambda_11) );
    q0 = vt::make(q_0[5], q_0[5] - mu, q_0[5] - mu); 
    if ( arePS(p0, q0) )
    { 
        //printf("failed by: co-plane 7\n"); 
        return false;   
    }

    return true;
}

}
}
 



/*int main(int argc, char const *argv[])
{
    typedef real3<float> vec_t;
    const auto int_tri = interset_triangles<float, vec_t>;



    //triangles for checking:
    
    //===intersecitons=== 

    //===no intersecitons===


    return 0;
}*/

#endif
