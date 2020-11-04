#include <cstdio>
#include <cmath>
#include <algorithm>


// aka float3
template<typename T>
class real3
{
public:    
    real3(T x_, T y_, T z_):
    x(x_), y(y_), z(z_)
    {

    }   
    ~real3()
    {

    }

    T x, y, z;

};

// FROM:
// https://github.com/kenny-evitt/three-js-triangle-triangle-collision-detection/blob/master/collision-tests.js


template<typename T>
bool areProjectionsSeparated(const T &p0, const T &p1, const T &p2, const T &q0, const T &q1, const T &q2)
{
    
    T min_p = std::min({p0, p1, p2});
    T max_p = std::max({p0, p1, p2});
    T min_q = std::min({q0, q1, q2});
    T max_q = std::max({q0, q1, q2});

    return ((min_p > max_q) || (max_p < min_q));

}

template<typename T, class Tv>
bool areProjectionsSeparated(const Tv &v0, const Tv &v1)
{
    T p0 = v0.x;
    T p1 = v0.y;
    T p2 = v0.z;
    T q0 = v1.x;
    T q1 = v1.y;
    T q2 = v1.z;

    return( areProjectionsSeparated<T>(p0, p1, p2, q0, q1, q2) );

}


//helper functions C-style
template<typename T, class Tv>
T dot_product(const Tv &v0, const Tv &v1)
{
    return(v0.x*v1.x+v0.y*v1.y+v0.z*v1.z);

}

template<typename T, class Tv>
Tv cross_product(const Tv &v0, const Tv &v1)
{
    T factor_i = v0.y*v1.z-v0.z*v1.y;
    T factor_j = v0.x*v1.z-v0.z*v1.x;
    T factor_k = v0.x*v1.y-v0.y*v1.x;

    Tv ret(factor_i,-factor_j,factor_k);
    return(ret);    
}


template<typename T, class Tv>
Tv add(const T &c0, const Tv &v0, const T &c1, const Tv &v1)
{
    return(Tv(c0*v0.x+c1*v1.x,c0*v0.y+c1*v1.y,c0*v0.z+c1*v1.z));
}

template<typename T, class Tv>
Tv sub_two(const Tv &v0, const Tv &v1)
{
    return add<T,Tv>(T(1), v0, T(-1), v1);
}

//helper functions ends


//main checker
template<typename T, class Tv>
bool interset_triangles(const Tv &t0_v0, const Tv &t0_v1, const Tv &t0_v2, const Tv &t1_v0, const Tv &t1_v1,const Tv &t1_v2)
{

    const auto sub = sub_two<T, Tv>;
    const auto cross = cross_product<T, Tv>;
    const auto dot = dot_product<T, Tv>;
    const auto arePS = areProjectionsSeparated<T,Tv>;


    Tv A0 = t0_v0;
    Tv A1 = t0_v1;
    Tv A2 = t0_v2;
    Tv E0 = sub(A1,A0);
    Tv E1 = sub(A2,A0);
    Tv E2 = sub(E1,E0); //A0? Original was E1-E0
    Tv N = cross(E0,E1); 

    Tv B0 = t1_v0;
    Tv B1 = t1_v1;
    Tv B2 = t1_v2;
    Tv F0 = sub(B1,B0);
    Tv F1 = sub(B2,B0);
    Tv F2 = sub(F1,F0); //B0? Original was F1-F0
    Tv M = cross(F0,F1);
    
    Tv D = sub(B0,A0);


//  Check axis N
    T NdotD = dot(N,D);
    Tv p0(0,0,0);
    Tv q0(NdotD, NdotD + dot(N,F0), NdotD + dot(N,F1) );
    if ( arePS(p0, q0) )
    {
        printf("failed by: 1\n");
        return false;      
    }
//  Check axis M
    T MdotD = dot(M,D);
    Tv p1(0, dot(M,E0), dot(M,E1) );
    Tv q1(MdotD, MdotD, MdotD);
    if ( arePS(p1, q1) )
    {
        printf("failed by: 2\n");
        return false;    
    }

//Check axis Ej \times Fk where {j,k}=0,1,2

    Tv Eall[3] = {E0, E1, E2};
    Tv Fall[3] = {F0, F1, F2};

    for(int j=0; j<3; j++)
    {
        T flag_j = (j==0?T(-1):T(1));

        for(int k=0; k<3;k++)
        {
            T flag_k = (k==0?T(1):T(-1));

            Tv E = Eall[j];
            Tv F = Fall[k];
            T ExFdD = dot(cross(E,F),D);
            Tv p2(0,0,flag_j*dot(N,F));
            Tv q2(ExFdD, ExFdD, ExFdD + flag_k*dot(M,E) );
            if ( arePS(p2, q2) )
            {
                printf("failed by: flag_j = %f, flag_k = %f, j=%i, k=%i\n",flag_j, flag_k, j, k);
                return false;    
            }
        }
    }
    
//testing co-planar triangles! 
//Error in the original method. Other axis are needed to seperate triangles for the co-planar case.
// l = ||N||; E_j X F_k = lambda_{j,k} N, j,k=0,1; F_0 X F_1 = mu N

    q0 = Tv(NdotD, NdotD, NdotD);
    if ( arePS(p0, q0) )
    {
        printf("failed by: co-plane 1\n");
        return false;      
    }

    T sqN = dot(N,N);

    T lambda_00 = dot(cross(E0, F0),N);
    T lambda_01 = dot(cross(E0, F1),N);
    T lambda_11 = dot(cross(E1, F1),N);
    T lambda_10 = dot(cross(E1, F0),N);

    T mu = dot(cross(F0, F1),N);

    //filling  (N X E_j, D) and (N X F_j, D), j = {0,1,2}
    T q_0[6];
    for(int j = 0; j<3; j++)
    {
        q_0[j] = dot(cross(N, Eall[j]),D);
        q_0[j+3] = dot(cross(N, Fall[j]),D);
    }

    p0 = Tv(T(0), T(0), sqN);
    q0 = Tv(q_0[0], q_0[0] + lambda_00, q_0[0] + lambda_01); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 2\n"); return false;   }

    p0 = Tv(T(0), -sqN, T(0));
    q0 = Tv(q_0[1], q_0[1] + lambda_10, q_0[1] + lambda_11); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 3\n"); return false;   }

    p0 = Tv(T(0), -sqN, -sqN);
    q0 = Tv(q_0[2], q_0[2] + (lambda_10-lambda_00), q_0[2] + (lambda_11-lambda_01)); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 4\n"); return false;   }

    p0 = Tv(T(0), -lambda_00, -lambda_10);
    q0 = Tv(q_0[3], q_0[3], q_0[3] + mu); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 5\n"); return false;   }

    p0 = Tv(T(0), -lambda_01, -lambda_11);
    q0 = Tv(q_0[4], q_0[4] - mu, q_0[4]); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 6\n"); return false;   }    

    p0 = Tv(T(0), (lambda_00-lambda_01), (lambda_10 - lambda_11) );
    q0 = Tv(q_0[5], q_0[5] - mu, q_0[5] - mu); 
    if ( arePS(p0, q0) ){ printf("failed by: co-plane 7\n"); return false;   }      

    return true;
}


 



int main(int argc, char const *argv[])
{
    typedef real3<float> float3;
    const auto int_tri = interset_triangles<float, float3>;

    // float3 t0_v0(1, 0, 0);
    // float3 t0_v1(0, 1, 0);
    // float3 t0_v2(0, 0, 1);


    //triangles for checking:
    
    //===intersecitons===
    //intersection by full segment:
    // float3 t1_v0(0.5, 0, 0); 
    // float3 t1_v1(0, 1.5, 0);
    // float3 t1_v2(0, 0, 1.5);

    //intersection by partial segment:
    // float3 t1_v0(0.7, 0, 0);
    // float3 t1_v1(1, 0.7, 0);
    // float3 t1_v2(1, 0.7, 0.2);

    //coincide:
    // float3 t1_v0(1, 0, 0);
    // float3 t1_v1(0, 1, 0);
    // float3 t1_v2(0, 0, 1);    

    //interseciton by side:
    // float3 t1_v0(1, 0, 0);
    // float3 t1_v1(0, 1, 0);
    // float3 t1_v2(0, 0, -1);

     //interseciton by point:
    // float3 t1_v0(1, 0, 0);
    // float3 t1_v1(0, -3, 0);
    // float3 t1_v2(0, 0, -3);   

    //intersection in one vertice and coplanar:
    // float3 t0_v0(1, 0, 1);
    // float3 t0_v1(0, 1, 1);
    // float3 t0_v2(0.5, 0, 1);

    // float3 t1_v0(-1, 0, 1);
    // float3 t1_v1(0, -1, 1);
    // float3 t1_v2(0.5, 0, 1);    

    //bad aspect ratio:
    // float3 t1_v0(0.1, 0.01, 0.001);
    // float3 t1_v1(100000000, 100000000, 10000000);
    // float3 t1_v2(100000000, -1000000000, -100000000); 

    //===no intersecitons===
     //no near interseciton by point:
    // float3 t1_v0(1+0.0000001, 0, 0);
    // float3 t1_v1(0, -3, 0);
    // float3 t1_v2(0, 0, -3); 

    //coplanar
    // float3 t1_v0(1+0.1, 0, 0);
    // float3 t1_v1(0, 1+0.1, 0);
    // float3 t1_v2(0, 0, 1+0.1);    

    //coplanar 2
    float3 t0_v0(1, 0, 1);
    float3 t0_v1(0, 1, 1);
    float3 t0_v2(0.5, 0, 1);

    float3 t1_v0(-1, 0, 1);
    float3 t1_v1(0, -1, 1);
    float3 t1_v2(-0.5, 0, 1); 

    //random
    // float3 t1_v0(1+0.1, 4, 2);
    // float3 t1_v1(1, 1+0.1, -1);
    // float3 t1_v2(2, 1, 1+0.1);   


    printf("test1:\n");
    printf("t0:\n");
    printf("%f %f %f\n", t0_v0.x, t0_v0.y, t0_v0.z);
    printf("%f %f %f\n", t0_v1.x, t0_v1.y, t0_v1.z);
    printf("%f %f %f\n", t0_v2.x, t0_v2.y, t0_v2.z);
    printf("t1:\n");
    printf("%f %f %f\n", t1_v0.x, t1_v0.y, t1_v0.z);
    printf("%f %f %f\n", t1_v1.x, t1_v1.y, t1_v1.z);
    printf("%f %f %f\n", t1_v2.x, t1_v2.y, t1_v2.z);


    if( int_tri(t0_v0, t0_v1, t0_v2, t1_v0, t1_v1, t1_v2) )
    {
        printf("\n Intersect \n");
    }

    t0_v0 = float3(4.509881019592285e+00,8.474619388580322e-01,-1.345000028610229e+00);
    t0_v1 = float3(4.849546432495117e+00,6.369206905364990e-01,-1.345000028610229e+00);
    t0_v2 = float3(4.849880695343018e+00,8.469204306602478e-01,-1.345000028610229e+00);
    t1_v0 = float3(4.985392570495605e+00,6.824411153793335e-01,-1.345000028610229e+00);
    t1_v1 = float3(5.025794029235840e+00,7.118983268737793e-01,-1.345000028610229e+00);
    t1_v2 = float3(4.885238170623779e+00,8.198057413101196e-01,-1.345000028610229e+00);

    printf("test2:\n");
    printf("t0:\n");
    printf("%f %f %f\n", t0_v0.x, t0_v0.y, t0_v0.z);
    printf("%f %f %f\n", t0_v1.x, t0_v1.y, t0_v1.z);
    printf("%f %f %f\n", t0_v2.x, t0_v2.y, t0_v2.z);
    printf("t1:\n");
    printf("%f %f %f\n", t1_v0.x, t1_v0.y, t1_v0.z);
    printf("%f %f %f\n", t1_v1.x, t1_v1.y, t1_v1.z);
    printf("%f %f %f\n", t1_v2.x, t1_v2.y, t1_v2.z);


    if( int_tri(t0_v0, t0_v1, t0_v2, t1_v0, t1_v1, t1_v2) )
    {
        printf("\n Intersect \n");
    }

    t0_v0 = float3(4.509881019592285e+00,-1.345000028610229e+00,8.474619388580322e-01);
    t0_v1 = float3(4.849546432495117e+00,-1.345000028610229e+00,6.369206905364990e-01);
    t0_v2 = float3(4.849880695343018e+00,-1.345000028610229e+00,8.469204306602478e-01);
    t1_v0 = float3(4.985392570495605e+00,-1.345000028610229e+00,6.824411153793335e-01);
    t1_v1 = float3(5.025794029235840e+00,-1.345000028610229e+00,7.118983268737793e-01);
    t1_v2 = float3(4.885238170623779e+00,-1.345000028610229e+00,8.198057413101196e-01);

    t0_v0 = float3(-1.345000028610229e+00,4.509881019592285e+00,8.474619388580322e-01);
    t0_v1 = float3(-1.345000028610229e+00,4.849546432495117e+00,6.369206905364990e-01);
    t0_v2 = float3(-1.345000028610229e+00,4.849880695343018e+00,8.469204306602478e-01);
    t1_v0 = float3(-1.345000028610229e+00,4.885238170623779e+00,8.198057413101196e-01);
    t1_v1 = float3(-1.345000028610229e+00,5.025794029235840e+00,7.118983268737793e-01);
    t1_v2 = float3(-1.345000028610229e+00,4.925639629364014e+00,8.492629528045654e-01); 

    printf("test3:\n");
    printf("t0:\n");
    printf("%f %f %f\n", t0_v0.x, t0_v0.y, t0_v0.z);
    printf("%f %f %f\n", t0_v1.x, t0_v1.y, t0_v1.z);
    printf("%f %f %f\n", t0_v2.x, t0_v2.y, t0_v2.z);
    printf("t1:\n");
    printf("%f %f %f\n", t1_v0.x, t1_v0.y, t1_v0.z);
    printf("%f %f %f\n", t1_v1.x, t1_v1.y, t1_v1.z);
    printf("%f %f %f\n", t1_v2.x, t1_v2.y, t1_v2.z);


    if( int_tri(t0_v0, t0_v1, t0_v2, t1_v0, t1_v1, t1_v2) )
    {
        printf("\n Intersect \n");
    }


    return 0;
}
