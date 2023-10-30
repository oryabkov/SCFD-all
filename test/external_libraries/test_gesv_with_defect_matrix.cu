
#include <iostream>
#include <random>
#include <scfd/utils/init_cuda.h>
#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/external_libraries/cusolver_wrap.h>
#include <scfd/external_libraries/regularize_qr_of_defect_matrix_cuda.h>
#include <scfd/external_libraries/regularize_qr_of_defect_matrix_cuda_impl.cuh>
#include <scfd/memory/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/array_nd_visible.h>
#include <scfd/arrays/first_index_fast_arranger.h>

#define TEST_DOUBLE

#ifdef TEST_FLOAT
using real = float;
#endif
#ifdef TEST_DOUBLE
using real = double;
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

void test_gesv_with_defect_matrix(matrix_t &mat, vector_t &b, vector_t &b_x, int defect)
{
    scfd::cublas_wrap cublas;
    scfd::cusolver_wrap cusolver(&cublas);

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
    thrust::copy(
        thrust::device_ptr<real>(b.array().raw_ptr()),
        thrust::device_ptr<real>(b.array().raw_ptr())+sz,
        thrust::device_ptr<real>(b_x.array().raw_ptr())
    );

    vector_t tau(sz);
    cusolver.geqrf(sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr());
    scfd::regularize_qr_of_defect_matrix_cuda(sz, mat_tmp.array().raw_ptr(), defect);
    cusolver.gesv_apply_qr(sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), b_x.array().raw_ptr());

    /*b_x.sync_from_array();
    for (int i = 0;i < sz;++i)
    {
        std::cout << b_x(i) << std::endl;
    }*/

    vector_t resid(sz);
    thrust::copy(
        thrust::device_ptr<real>(b.array().raw_ptr()),
        thrust::device_ptr<real>(b.array().raw_ptr())+sz,
        thrust::device_ptr<real>(resid.array().raw_ptr())
    );
    cublas.gemv('N', sz, mat.array().raw_ptr(), sz, sz, real(1), b_x.array().raw_ptr(), real(-1), resid.array().raw_ptr());
    
    //calc norm
    resid.sync_from_array();
    b.sync_from_array();
    real resid_norm(0), b_norm(0);
    for (int i = 0;i < sz;++i)
    {
        resid_norm += resid(i)*resid(i);
        b_norm += b(i)*b(i);
    }
    resid_norm = std::sqrt(resid_norm);
    b_norm = std::sqrt(b_norm);

    std::cout << "resid_norm = " << resid_norm << std::endl;
    std::cout << "b_norm = " << b_norm << std::endl;
}

void test_gesv_with_defect_matrix_with_init(
    std::initializer_list<std::initializer_list<real>> mat_vals,
    std::initializer_list<real> rhs_vals,
    int defect)
{
    matrix_t mat;
    init_array_matrix_with_values(mat, mat_vals);
    vector_t b,x(rhs_vals.size());
    init_array_vector_with_values(b, rhs_vals);

    test_gesv_with_defect_matrix(mat, b, x, defect);
}

void test_gesv_with_defect_matrix_with_1d_poisson(int sz)
{
    matrix_t mat(sz,sz);
    vector_t b(sz), x(sz);

    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            mat(i,j) = real(0);
        }

        if (i-1>=0)
        {
            mat(i,i-1) = real(1);
        }
        if (i+1<sz)
        {
            mat(i,i+1) = real(1);
        }
        mat(i,i) = real(-2);
    }
    mat(sz-1,0) = real(1);
    mat(0,sz-1) = real(1);
    mat.sync_to_array();

    std::mt19937 gen(1);
    std::uniform_real_distribution<real> dis(0., 1.);
    real sum_b(0);
    for (int i = 0;i < sz;++i)
    {
        b(i) = dis(gen);
        sum_b += b(i);
    }
    for (int i = 0;i < sz;++i)
    {
        b(i) = b(i) - sum_b/sz;
    }
    b.sync_to_array();

    test_gesv_with_defect_matrix(mat, b, x, 0);

    x.sync_from_array();
    real sum_x(0);
    for (int i = 0;i < sz;++i)
    {
        sum_x += x(i);
    }
    std::cout << "sum_x = " << sum_x << std::endl;
}

int main(int argc, char const *args[])
{
    scfd::utils::init_cuda_persistent();

    std::cout << "test zero 4x4 matrix with zero rhs" << std::endl;
    test_gesv_with_defect_matrix_with_init(
        {{ 0, 0, 0, 0 },
         { 0, 0, 0, 0 },
         { 0, 0, 0, 0 },
         { 0, 0, 0, 0 }},
        {  0,
           0,
           0,
           0 },
        4  
    );

    std::cout << "test deg 4x4 matrix with rank=2 and rhs in image" << std::endl;
    test_gesv_with_defect_matrix_with_init(
        {{ 1, 2, 3,  4 },
         { 5, 6, 7,  8 },
         { 9,10, 11,12 },
         { 13,14,15,16 }},
        {  30,
           70,
          110,
          150 },
        2  
    );

    std::cout << "test deg 1000x1000 posiion 1d matrix with defect=1 and rhs in image" << std::endl;
    test_gesv_with_defect_matrix_with_1d_poisson(2000);

    return 0;
}