
#include <iostream>
#include <limits>
#include <random>
#include "gtest/gtest.h"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include <scfd/external_libraries/cusolver_wrap_singleton_impl.h>
#include <scfd/memory/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/array_nd_visible.h>
#include <scfd/arrays/first_index_fast_arranger.h>

#define MLUPS 50
/// Change MLUPS for some small value to test that test actually tests something
/// #define MLUPS 1
//#define TEST_FLOAT
#define TEST_DOUBLE

#ifdef TEST_FLOAT
using real = float;
#define EXPECT_REAL_EQ EXPECT_FLOAT_EQ
#endif
#ifdef TEST_DOUBLE
using real = double;
#define EXPECT_REAL_EQ EXPECT_DOUBLE_EQ
#endif
using mem_t = scfd::memory::cuda_device;
template<scfd::arrays::ordinal_type... Dims>
using col_major_arranger_t = scfd::arrays::first_index_fast_arranger<Dims...>;
using matrix_t = scfd::arrays::array_nd_visible<real,2,mem_t>;
using vector_t = scfd::arrays::array_nd_visible<real,1,mem_t>;

void init_array_matrix_with_values(matrix_t &mat, std::initializer_list<std::initializer_list<real>> vals)
{
    int rows = vals.size();
    int cols = 0;
    for (auto it = vals.begin();it != vals.end();++it)
    {
        if (it == vals.begin())
        {
            cols = it->size();
        }
        else
        {
            if (cols != it->size())
            {
                throw std::logic_error("init_array_matrix_with_values: init of nonuniform size");
            }
        }
    }

    mat.init(rows, cols);

    int row = 0;
    for (auto row_it = vals.begin();row_it != vals.end();++row_it,++row)
    {
        int col = 0;
        for (auto it = row_it->begin();it != row_it->end();++it,++col)
        {
            mat(row,col) = *it;
        }
    }

    mat.sync_to_array();
}

void init_array_vector_with_values(vector_t &vec, std::initializer_list<real> vals)
{
    int sz = vals.size();

    vec.init(sz);

    int i = 0;
    for (auto it = vals.begin();it != vals.end();++it,++i)
    {
        vec(i) = *it;
    }

    vec.sync_to_array();
}

real calc_fro(matrix_t &mat)
{
    mat.sync_from_array();
    real res(0);
    int sz1 = mat.size_nd()[0],
        sz2 = mat.size_nd()[1];
    for (int i = 0;i < sz1;++i)
    {
        for (int j = 0;j < sz2;++j)
        {
            real abs_v = std::abs(mat(i,j));
            res += abs_v*abs_v;
        }
    }
    res = std::sqrt(res);
    return res;
}

real calc_r2_norm(vector_t &vec)
{
    vec.sync_from_array();
    real res(0);
    int sz = vec.size();
    for (int i = 0;i < sz;++i)
    {
        real abs_v = std::abs(vec(i));
        res += abs_v*abs_v;
    }
    res = std::sqrt(res);
    return res;
}

real calc_eps(real val1, real val2, real fro)
{
    return std::max(std::max(std::abs(val1),std::abs(val2)),fro)*std::numeric_limits<real>::epsilon()*MLUPS;
}

void test_gesv(
    matrix_t &mat, vector_t &rhs, vector_t &ref_x, bool use_apply_qr, bool check_refs
)
{
    scfd::cublas_wrap &cublas = scfd::cublas_wrap::inst();
    scfd::cusolver_wrap &cusolver = scfd::cusolver_wrap::inst();

    if (mat.size_nd()[0] != mat.size_nd()[1])
    {
        throw std::logic_error("test_geqrf_orgqr: matrix is not square");
    }
    int sz = mat.size_nd()[0];

    matrix_t mat_tmp(sz,sz);
    thrust::copy(
        thrust::device_ptr<real>(mat.array().raw_ptr()),
        thrust::device_ptr<real>(mat.array().raw_ptr())+sz*sz,
        thrust::device_ptr<real>(mat_tmp.array().raw_ptr())
    );
    vector_t x(sz);
    thrust::copy(
        thrust::device_ptr<real>(rhs.array().raw_ptr()),
        thrust::device_ptr<real>(rhs.array().raw_ptr())+sz,
        thrust::device_ptr<real>(x.array().raw_ptr())
    );
    if (use_apply_qr)
    {
        vector_t tau(sz);
        cusolver.geqrf(sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr());
        cusolver.gesv_apply_qr(sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), x.array().raw_ptr());
    }
    else
    {
        cusolver.gesv(sz, mat_tmp.array().raw_ptr(), x.array().raw_ptr());
    }

    x.sync_from_array();
    if (check_refs)
    {
        ref_x.sync_from_array();
        real ref_r2_norm = calc_r2_norm(ref_x);
        for (int i = 0;i < sz;++i)
        {
            real eps = calc_eps(ref_x(i), x(i), ref_r2_norm);
            EXPECT_NEAR(ref_x(i), x(i), eps);
        }
    }

    vector_t mat_mul_x(sz);
    cublas.gemv('N', sz, mat.array().raw_ptr(), sz, sz, real(1), x.array().raw_ptr(), real(0), mat_mul_x.array().raw_ptr());
    mat_mul_x.sync_from_array();
    rhs.sync_from_array();
    real rhs_r2_norm = calc_r2_norm(rhs);
    for (int i = 0;i < sz;++i)
    {
        real eps = calc_eps(rhs(i), mat_mul_x(i), rhs_r2_norm);
        EXPECT_NEAR(rhs(i), mat_mul_x(i), eps);
    }
}

void test_gesv_with_init(
    std::initializer_list<std::initializer_list<real>> mat_vals,
    std::initializer_list<real> rhs_vals,
    std::initializer_list<real> ref_x_vals,
    bool use_apply_qr, bool check_refs
)
{
    matrix_t mat;
    vector_t rhs, ref_x;
    init_array_matrix_with_values(mat, mat_vals);
    init_array_vector_with_values(rhs, rhs_vals);
    if (check_refs)
    {
        init_array_vector_with_values(ref_x, ref_x_vals);
    }
    test_gesv(mat, rhs, ref_x, use_apply_qr, check_refs);
}

void test_gesv_with_rand(int sz, bool use_apply_qr)
{
    matrix_t mat(sz,sz);
    vector_t rhs(sz), ref_x;
    std::mt19937 gen(1);
    std::uniform_real_distribution<real> dis(-10., 10.);
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            mat(i,j) = dis(gen);
        }
        rhs(i) = dis(gen);
    }
    mat.sync_to_array();
    rhs.sync_to_array();
    test_gesv(mat, rhs, ref_x, use_apply_qr, false);
}

TEST(CusolverWrapTest, GeSVIdent3x3Mat) 
{
    test_gesv_with_init(
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
        {1,
         1,
         1},
        {1,
         1,
         1},
         false, true);
}

TEST(CusolverWrapTest, GeSVApplyQRIdent3x3Mat) 
{
    test_gesv_with_init(
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
        {1,
         1,
         1},
        {1,
         1,
         1},
         true, true);
}

TEST(CusolverWrapTest, GeSVDiag4x4Mat) 
{
    test_gesv_with_init(
        {{ 1, 0, 0, 0},
         { 0,-2, 0, 0},
         { 0, 0, 3, 0},
         { 0, 0, 0,-3}},
        {  1,
           2,
           3,
           3},
        {  1,
          -1,
           1,
          -1},
         false, true);
}

TEST(CusolverWrapTest, GeSVApplyQRDiag4x4Mat) 
{
    test_gesv_with_init(
        {{ 1, 0, 0, 0},
         { 0,-2, 0, 0},
         { 0, 0, 3, 0},
         { 0, 0, 0,-3}},
        {  1,
           2,
           3,
           3},
        {  1,
          -1,
           1,
          -1},
         true, true);
}

/// Generated with Matlab
TEST(CusolverWrapTest, GeSVGen5x5Mat) 
{
    test_gesv_with_init(
        {{ 1, 0, 2, 0, 0},
         { 0, 2, 0, 0, 1},
         { 1, 0, 3, 0, 1},
         { 1, 0, 0,-3, 1},
         {10, 0, 0, 0,-5}},
        {  7,
           9,
          15,
          -6,
         -15},
        {  1,
           2,
           3,
           4,
           5},
         false, true);
}

/// Generated with Matlab
TEST(CusolverWrapTest, GeSVApplyQRGen5x5Mat) 
{
    test_gesv_with_init(
        {{ 1, 0, 2, 0, 0},
         { 0, 2, 0, 0, 1},
         { 1, 0, 3, 0, 1},
         { 1, 0, 0,-3, 1},
         {10, 0, 0, 0,-5}},
        {  7,
           9,
          15,
          -6,
         -15},
        {  1,
           2,
           3,
           4,
           5},
         true, true);
}

TEST(CusolverWrapTest, GeSVFull99x99RandMat) 
{
    test_gesv_with_rand(99, false);
}

TEST(CusolverWrapTest, GeSVFull114x114RandMat) 
{
    test_gesv_with_rand(114, false);
}

TEST(CusolverWrapTest, GeSVFull200x200RandMat) 
{
    test_gesv_with_rand(200, false);
}

TEST(CusolverWrapTest, GeSVApplyQRFull99x99RandMat) 
{
    test_gesv_with_rand(99, true);
}

TEST(CusolverWrapTest, GeSVApplyQRFull114x114RandMat) 
{
    test_gesv_with_rand(114, true);
}

TEST(CusolverWrapTest, GeSVApplyQRFull200x200RandMat) 
{
    test_gesv_with_rand(200, true);
}

void test_geqrf_orgqr(
    matrix_t &mat, matrix_t &q_ref_mat, matrix_t &r_ref_mat,
    bool check_refs
)
{
    scfd::cublas_wrap &cublas = scfd::cublas_wrap::inst();
    scfd::cusolver_wrap &cusolver = scfd::cusolver_wrap::inst();

    if (mat.size_nd()[0] != mat.size_nd()[1])
    {
        throw std::logic_error("test_geqrf_orgqr: matrix is not square");
    }
    int sz = mat.size_nd()[0];
    matrix_t inital_mat(sz,sz), q_mat(sz,sz), q_mul_r_mat(sz,sz);
    vector_t tau(sz);
    /// Fill with some trash to test
    //init_array_vector_with_values(tau, {-10,-20,-30});

    thrust::copy(
        thrust::device_ptr<real>(mat.array().raw_ptr()),
        thrust::device_ptr<real>(mat.array().raw_ptr())+sz*sz,
        thrust::device_ptr<real>(inital_mat.array().raw_ptr())
    );

    cusolver.geqrf(sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr());
    cusolver.orgqr(sz, sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
    mat.sync_from_array();
    q_mat.sync_from_array();
    tau.sync_from_array();
    if (check_refs)
    {
        /// Check R part only of mat
        real r_ref_mat_fro = calc_fro(r_ref_mat);
        for (int i = 0;i < sz;++i)
        {
            for (int j = 0;j < sz;++j)
            {
                if (j >= i)
                {
                    real eps = calc_eps(r_ref_mat(i,j), mat(i,j), r_ref_mat_fro);
                    EXPECT_NEAR(r_ref_mat(i,j), mat(i,j), eps);
                }
            }
        }
        /// Check Q matrix
        real q_ref_mat_fro = calc_fro(q_ref_mat);
        for (int i = 0;i < sz;++i)
        {
            for (int j = 0;j < sz;++j)
            {
                real eps = calc_eps(q_ref_mat(i,j), q_mat(i,j), q_ref_mat_fro);
                EXPECT_NEAR(q_ref_mat(i,j), q_mat(i,j), eps);
            }
        }
    }
    /*std::cout << "a_mat (Householder vectors of Q in lower part and R matrix in upper part)" << std::endl;
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            std::cout << mat(i,j) << " ";
        }
        std::cout << " : " << tau(i) << std::endl;
    }
    std::cout << "q_mat" << std::endl;
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            std::cout << q_mat(i,j) << " ";
        }
        std::cout << std::endl;
    }*/

    /// Create pure R matrix
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            if (j < i)
            {
                mat(i,j) = 0.;
            }
        }
    }
    mat.sync_to_array();

    cublas.gemm('N', 'N', sz, sz, sz, real(1), q_mat.array().raw_ptr(), sz, mat.array().raw_ptr() , sz, real(0), q_mul_r_mat.array().raw_ptr(), sz);
    q_mul_r_mat.sync_from_array();

    /// Check QR vs initial matrix
    real inital_mat_fro = calc_fro(inital_mat);
    inital_mat.sync_from_array();
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            real eps = calc_eps(inital_mat(i,j), q_mul_r_mat(i,j), inital_mat_fro);
            EXPECT_NEAR(inital_mat(i,j), q_mul_r_mat(i,j), eps);
        }
    }

    //cusolver.gesv_apply_qr(sz, mat.array().raw_ptr(), tau.array().raw_ptr(), T* b_x);
}

void test_geqrf_orgqr_with_init(
    std::initializer_list<std::initializer_list<real>> mat_vals,
    std::initializer_list<std::initializer_list<real>> q_ref_mat_vals,
    std::initializer_list<std::initializer_list<real>> r_ref_mat_vals,
    bool check_refs
)
{
    matrix_t mat, q_ref_mat, r_ref_mat;
    init_array_matrix_with_values(mat, mat_vals);
    if (check_refs)
    {
        init_array_matrix_with_values(q_ref_mat, q_ref_mat_vals);
        init_array_matrix_with_values(r_ref_mat, r_ref_mat_vals);
    }
    test_geqrf_orgqr(mat, q_ref_mat, r_ref_mat, check_refs);
}

void test_geqrf_orgqr_with_rand(int sz)
{
    matrix_t mat(sz,sz), q_ref_mat, r_ref_mat;
    std::mt19937 gen(1);
    std::uniform_real_distribution<real> dis(-10., 10.);
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            mat(i,j) = dis(gen);
        }
    }
    mat.sync_to_array();
    test_geqrf_orgqr(mat, q_ref_mat, r_ref_mat, false);
}

TEST(CusolverWrapTest, GeQRFOrGQRIdent3x3Mat) 
{
    //scfd::cublas_wrap cublas1;
    test_geqrf_orgqr_with_init(
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
         true);
}

TEST(CusolverWrapTest, GeQRFOrGQRZero3x3Mat) 
{
    test_geqrf_orgqr_with_init(
        {{0,0,0},
         {0,0,0},
         {0,0,0}},
        {{1,0,0},
         {0,1,0},
         {0,0,1}},
        {{0,0,0},
         {0,0,0},
         {0,0,0}},
         true);
}

TEST(CusolverWrapTest, GeQRFOrGQRDiag5x5Mat) 
{
    test_geqrf_orgqr_with_init(
        {{1, 0, 0, 0, 0 },
         {0,-1, 0, 0, 0 },
         {0, 0, 2, 0, 0 },
         {0, 0, 0, 3, 0 },
         {0, 0, 0, 0, 10}},
        {{1,0,0,0,0},
         {0,1,0,0,0},
         {0,0,1,0,0},
         {0,0,0,1,0},
         {0,0,0,0,1}},
        {{1, 0, 0, 0, 0 },
         {0,-1, 0, 0, 0 },
         {0, 0, 2, 0, 0 },
         {0, 0, 0, 3, 0 },
         {0, 0, 0, 0, 10}},
         true);
}

TEST(CusolverWrapTest, GeQRFOrGQRDiagDeg5x5Mat) 
{
    test_geqrf_orgqr_with_init(
        {{1, 0, 0, 0, 0 },
         {0,-1, 0, 0, 0 },
         {0, 0, 0, 0, 0 },
         {0, 0, 0, 0, 0 },
         {0, 0, 0, 0, 10}},
        {{1,0,0,0,0},
         {0,1,0,0,0},
         {0,0,1,0,0},
         {0,0,0,1,0},
         {0,0,0,0,1}},
        {{1, 0, 0, 0, 0 },
         {0,-1, 0, 0, 0 },
         {0, 0, 0, 0, 0 },
         {0, 0, 0, 0, 0 },
         {0, 0, 0, 0, 10}},
         true);
}

/// Here matlab qr result is given as reference but
/// initial matrix is degenerate so expected Q matrix values differs from Matlab reference
TEST(CusolverWrapTest, GeQRFOrGQRFullDeg4x4Mat) 
{
    test_geqrf_orgqr_with_init(
        {{ 1, 2, 3,  4 },
         { 5, 6, 7,  8 },
         { 9,10, 11,12 },
         { 13,14,15,16 }},
        {{-6.019292654288466e-02, -8.344919481901565e-01, -1.989674679255930e-01, -5.103057384620305e-01},
         {-3.009646327144230e-01, -4.576246167494406e-01, 6.456493974038999e-01, 5.321060567518290e-01},
         {-5.417363388859614e-01, -8.075728530872478e-02, -6.943963910310217e-01, 4.667051018824332e-01},
         {-7.825080450574998e-01, 2.961100461319910e-01, 2.477144615527145e-01, -4.885054201722319e-01}},
        {{-1.661324772583615e+01, -1.829864966903692e+01, -1.998405161223769e+01, -2.166945355543846e+01},
         {0.000000000000000e+00, -1.076763804116331e+00, -2.153527608232658e+00, -3.230291412348994e+00},
         {0.000000000000000e+00, 0.000000000000000e+00, 1.691041330490230e-15, -2.915588500845223e-16},
         {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.574417790456421e-15}},
         false);
    //std::cout << "end" << std::endl;
}

/// Here matlab qr result is given as reference
TEST(CusolverWrapTest, GeQRFOrGQRFullNonDeg4x4Mat) 
{
    test_geqrf_orgqr_with_init(
        {{ 1, 2, 3,  4 },
         { 5, 6, 7,  9 },
         { 9,10, 11,12 },
         { 13,13,15,16 }},
        {{-6.019292654288466e-02,-6.397750659233958e-01,6.483810684699627e-01,-4.082482904638634e-01},
         {-3.009646327144230e-01,-4.927003381249160e-01,7.771561172376096e-16,8.164965809277256e-01},
         {-5.417363388859614e-01,-3.456256103264341e-01,-6.483810684699606e-01,-4.082482904638635e-01},
         {-7.825080450574998e-01,4.779928653450678e-01,3.990037344430521e-01,5.551115123125783e-16}},
        {{-1.661324772583615e+01,-1.751614162397941e+01,-1.998405161223769e+01,-2.197041818815288e+01},
         {0.000000000000000e+00,-1.478101014374743e+00,-2.000216298059352e+00,-3.493024785213949e+00},
         {0.000000000000000e+00,0.000000000000000e+00,7.980074688861096e-01,1.197011203329168e+00},
         {0.000000000000000e+00,0.000000000000000e+00,0.000000000000000e+00,8.164965809277238e-01}},
         //here one digit is changed for incorrect one; uncomment to test that test actually tests something
         //{0.000000000000000e+00,0.000000000000000e+00,0.000000000000000e+00,8.264965809277238e-01}},
         true);
    //std::cout << "end" << std::endl;
}

TEST(CusolverWrapTest, GeQRFOrGQRFull99x99RandMat) 
{
    test_geqrf_orgqr_with_rand(99);
}

TEST(CusolverWrapTest, GeQRFOrGQRFull114x114RandMat) 
{
    test_geqrf_orgqr_with_rand(114);
}

TEST(CusolverWrapTest, GeQRFOrGQRFull200x200RandMat) 
{
    test_geqrf_orgqr_with_rand(200);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    scfd::utils::init_cuda_persistent();

    scfd::cublas_wrap cublas(true);
    scfd::cusolver_wrap cusolver(&cublas,true);
    //scfd::cublas_wrap::set_inst(&cublas);
    //scfd::cusolver_wrap::set_inst(&cusolver);

    return RUN_ALL_TESTS();
}
