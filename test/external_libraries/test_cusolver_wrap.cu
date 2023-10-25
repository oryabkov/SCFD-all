
#include <iostream>
#include "gtest/gtest.h"
#include <scfd/utils/init_cuda.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include <scfd/memory/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/array_nd_visible.h>
#include <scfd/arrays/first_index_fast_arranger.h>

#define TEST_DOUBLE

#ifdef TEST_FLOAR
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

void test_geqrforgqr(
    std::initializer_list<std::initializer_list<real>> mat_vals,
    std::initializer_list<std::initializer_list<real>> q_ref_mat_vals,
    std::initializer_list<std::initializer_list<real>> r_ref_mat_vals
)
{
    scfd::cublas_wrap cublas;
    scfd::cusolver_wrap cusolver(&cublas);

    init_array_matrix_with_values(mat, mat_vals);
    if (mat.size_nd()[0] != mat.size_nd()[1])
    {
        throw std::logic_error("test_geqrforgqr: matrix is not square");
    }
    int sz = mat.size_nd()[0];
    vector_t tau(sz);
    /// Fill with some trash to test
    //init_array_vector_with_values(tau, {-10,-20,-30});

    cusolver.geqrf(sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr());
    cusolver.orgqr(sz, sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
    mat.sync_from_array();
    q_mat.sync_from_array();
    tau.sync_from_array();
    /// Check R part only of mat
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            if (j >= i)
            {
                EXPECT_REAL_EQ((i==j?real(1):real(0)), mat(i,j));
            }
        }
    }
    /// Check Q matrix
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            EXPECT_REAL_EQ((i==j?real(1):real(0)), q_mat(i,j));
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

    //cusolver.gesv_apply_qr(sz, mat.array().raw_ptr(), tau.array().raw_ptr(), T* b_x);
}

TEST(CusolverWrapTest, GeQRFOrGQRIdent3x3Mat) 
{
    scfd::cublas_wrap cublas;
    scfd::cusolver_wrap cusolver(&cublas);

    int sz = 3;
    matrix_t mat, q_mat(sz,sz);
    init_array_matrix_with_values(
        mat, 
        {{1,0,0},
         {0,1,0},
         {0,0,1}}
    );
    mat.sync_to_array();
    vector_t tau;
    /// Fill with some trash to test
    init_array_vector_with_values(tau, {-10,-20,-30});

    cusolver.geqrf(sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr());
    cusolver.orgqr(sz, sz, sz, mat.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
    mat.sync_from_array();
    q_mat.sync_from_array();
    tau.sync_from_array();
    /// Check R part only of mat
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            if (j >= i)
            {
                EXPECT_REAL_EQ((i==j?real(1):real(0)), mat(i,j));
            }
        }
    }
    /// Check Q matrix
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            EXPECT_REAL_EQ((i==j?real(1):real(0)), q_mat(i,j));
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

    //cusolver.gesv_apply_qr(sz, mat.array().raw_ptr(), tau.array().raw_ptr(), T* b_x);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    scfd::utils::init_cuda_persistent();

    return RUN_ALL_TESTS();
}
