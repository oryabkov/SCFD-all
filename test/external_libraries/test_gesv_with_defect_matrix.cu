
#include <iostream>
#include <random>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/utils/system_timer_event.h>
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
template<scfd::arrays::ordinal_type... Dims>
using row_major_arranger_t = scfd::arrays::last_index_fast_arranger<Dims...>;
using matrix_t = scfd::arrays::array_nd_visible<real,2,mem_t,col_major_arranger_t>;
using row_major_matrix_t = scfd::arrays::array_nd_visible<real,2,mem_t,row_major_arranger_t>;
using vector_t = scfd::arrays::array_nd_visible<real,1,mem_t>;
using ptr_vector_t = scfd::arrays::array_nd_visible<real*,1,mem_t>;
using int_vector_t = scfd::arrays::array_nd_visible<int,1,mem_t>;

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

///NOTE Only! copies only host image of the matrix to host image of another matrix
template<class MatFrom, class MatTo>
void copy_to_another_layout(MatFrom &mat, MatTo &new_mat)
{
    int sz = mat.size_nd()[0];
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            new_mat(i,j) = mat(i,j);
        }
    }
}

template<class T>
__global__ void ker_r_inv_matrix_set_ident(
    int n, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < n*n)) return;
    int i = ind%n,
        j = ind/n;
    r_inv_mat[i+j*n] = (i == j? real(1) : real(0));
}

template<class T>
__global__ void ker_precalc_invert_r_diag(
    int n, T *mat_tmp, T *r_diag_inv
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < n)) return;
    r_diag_inv[i] = real(1)/mat_tmp[i*n+i];
}

template<class T>
__global__ void ker_invert_r_diag(
    int n, const T *r_diag_inv, T *mat_tmp, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < n*n)) return;
    int i = ind%n,
        j = ind/n;
    mat_tmp[i+j*n] *= r_diag_inv[i];
    if (i == j)
    {
        r_inv_mat[i+j*n] *= r_diag_inv[i];
    }
}

template<class T>
__global__ void ker_copy_mat_tmp_col_i(
    int n, int i, const T *mat_tmp, T *mat_tmp_col_i
)
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i1 < i)) return;
    mat_tmp_col_i[i1] = mat_tmp[i1+i*n];
}

template<class T>
__global__ void ker_back_elimination(
    int n, int i, const T *mat_tmp_col_i, T *r_inv_mat
)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ind < (n-i)*i)) return;
    int i1 = ind%i,
        j = i + ind/i;

    //make mat_tmp(i1,i) to be 0
    real mul = mat_tmp_col_i[i1];
    r_inv_mat[i1+j*n] -= r_inv_mat[i+j*n]*mul;
}

void test_gesv_with_defect_matrix(matrix_t &mat, vector_t &b, vector_t &b_x, int defect, int solve_algo)
{
    scfd::cublas_wrap cublas;
    scfd::cusolver_wrap cusolver(&cublas);

    if (mat.size_nd()[0] != mat.size_nd()[1])
    {
        throw std::logic_error("test_geqrf_orgqr: matrix is not square");
    }

    int sz = mat.size_nd()[0];

    matrix_t mat_tmp(sz,sz), q_mat(sz,sz), r_inv_mat(sz,sz);
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

    scfd::utils::cuda_timer_event  t1_refactor,t2_refactor;
    t1_refactor.record();
    vector_t tau(sz);
    cusolver.geqrf(sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr());
    scfd::regularize_qr_of_defect_matrix_cuda(sz, mat_tmp.array().raw_ptr(), defect);
    t2_refactor.record();
    std::cout << "common refactor time: " << t2_refactor.elapsed_time(t1_refactor) << "ms" << std::endl;

    scfd::utils::cuda_timer_event  t1,t2;
    //VERSION0: using gesv by QR geqrf representation
    if (solve_algo == 0)
    {
        t1.record();
        cusolver.gesv_apply_qr(sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), b_x.array().raw_ptr());
        t2.record();
        std::cout << "solve time: " << t2.elapsed_time(t1) << "ms" << std::endl;
    }
    
    //VERSION1: get explicit Q, explicit R inversion, then apply using cublas
    if (solve_algo == 1) 
    {
        //get Q matrix explicitly
        cusolver.orgqr(sz, sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
        mat_tmp.sync_from_array();
        /// Init r_inv with ident matrix
        scfd::utils::system_timer_event host_t1,host_t2;
        host_t1.record();
        for (int i = 0;i < sz;++i)
        {
            for (int j = 0;j < sz;++j)
            {
                r_inv_mat(i,j) = (i == j? real(1) : real(0));
            }
        }
        //row_major_matrix_t mat_tmp_row_major(sz,sz), r_inv_mat_row_major(sz,sz);
        // copy_to_another_layout(mat_tmp, mat_tmp_row_major);
        // copy_to_another_layout(r_inv_mat, r_inv_mat_row_major);
        // for (int i = 0;i < sz;++i)
        // {
        //     //make mat_tmp(i,i) to be 1
        //     real diag = mat_tmp_row_major(i,i);
        //     #pragma omp parallel for shared(mat_tmp_row_major,r_inv_mat_row_major)
        //     for (int j = 0;j < sz;++j)
        //     {
        //         mat_tmp_row_major(i,j) /= diag;
        //         r_inv_mat_row_major(i,j) /= diag;
        //     }
        //     for (int i1 = 0;i1 < i;++i1)
        //     {
        //         //make mat_tmp(i1,i) to be 0
        //         real mul = mat_tmp_row_major(i1,i);
        //         #pragma omp parallel for shared(mat_tmp_row_major,r_inv_mat_row_major)
        //         for (int j = 0;j < sz;++j)
        //         {
        //             mat_tmp_row_major(i1,j) -= mat_tmp_row_major(i,j)*mul;
        //             r_inv_mat_row_major(i1,j) -= r_inv_mat_row_major(i,j)*mul;
        //         }
        //     }
        // }
        // copy_to_another_layout(mat_tmp_row_major, mat_tmp);
        // copy_to_another_layout(r_inv_mat_row_major, r_inv_mat);
        //make mat_tmp(i,i) to be 1
        //std::cout << "here1" << std::endl;
        real r_diag_inv[sz];
        for (int i = 0;i < sz;++i)
        {
            r_diag_inv[i] = real(1)/mat_tmp(i,i);
        }
        //std::cout << "here2" << std::endl;
        for (int j = 0;j < sz;++j)
        {
            for (int i = 0;i <= j;++i)
            {
                mat_tmp(i,j) *= r_diag_inv[i];
            }
            r_inv_mat(j,j) *= r_diag_inv[j];
        }
        //std::cout << "here3" << std::endl;
        for (int i = sz-1;i >= 0;--i)
        {
            real mat_tmp_col_i[i];
            for (int i1 = 0;i1 < i;++i1)
            {
                mat_tmp_col_i[i1] = mat_tmp(i1,i);
            }
            //std::cout << "here4: i = " << i << std::endl;
            #pragma omp parallel for
            for (int j = i;j < sz;++j)
            {
                for (int i1 = 0;i1 < i;++i1)
                {
                    //make mat_tmp(i1,i) to be 0
                    real mul = mat_tmp_col_i[i1];
                    //mat_tmp(i1,j) -= mat_tmp(i,j)*mul;
                    r_inv_mat(i1,j) -= r_inv_mat(i,j)*mul;
                }
            }
            /*std::cout << "i = " << i << std::endl;
            for (int ii = 0;ii < i;++ii)
            {
                std::cout << mat_tmp_col_i[ii] << std::endl;
            }*/
            /*for (int ii = 0;ii < sz;++ii)
            {
                for (int jj = 0;jj < sz;++jj)
                {
                    std::cout << r_inv_mat(ii,jj) << " ";
                }
                std::cout << std::endl;
            }*/
        }
        //mat_tmp.sync_to_array();
        r_inv_mat.sync_to_array();
        host_t2.record();
        std::cout << "invert time: " << host_t2.elapsed_time(host_t1) << "ms" << std::endl;
        vector_t tmp(sz);
        t1.record();
        cublas.gemv('T', sz, q_mat.array().raw_ptr(), sz, sz, real(1), b_x.array().raw_ptr(), real(0), tmp.array().raw_ptr());
        cublas.gemv('N', sz, r_inv_mat.array().raw_ptr(), sz, sz, real(1), tmp.array().raw_ptr(), real(0), b_x.array().raw_ptr());    
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t2.record();
        std::cout << "solve time: " << t2.elapsed_time(t1) << "ms" << std::endl;
    }

    //VERSION2: get explicit Q, explicit R inversion using cublas, then apply using cublas
    if (solve_algo == 2)
    {
        //get Q matrix explicitly
        cusolver.orgqr(sz, sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
        scfd::utils::cuda_timer_event t1_1,t2_1;
        t1_1.record();
        ptr_vector_t a_array(1), c_array(1);
        /// Vanish lower part of matrix mat_tmp so that mat_tmp contains clean r part of QR matrix factorization
        mat_tmp.sync_from_array();
        for (int i = 0;i < sz;++i)
        {
            for (int j = 0;j < sz;++j)
            {
                if (j >= i) continue;
                mat_tmp(i,j) = real(0);
            }
        }
        mat_tmp.sync_to_array();
        a_array(0) = mat_tmp.array().raw_ptr();
        a_array.sync_to_array();
        c_array(0) = r_inv_mat.array().raw_ptr();
        c_array.sync_to_array();
        int_vector_t info_array(1);
        CUBLAS_SAFE_CALL( cublasDgetriBatched(*cublas.get_handle(), sz, a_array.array().raw_ptr(), sz, nullptr, c_array.array().raw_ptr(), sz, info_array.array().raw_ptr(), 1) );
        t2_1.record();
        std::cout << "invert time: " << t2_1.elapsed_time(t1_1) << "ms" << std::endl;
        vector_t tmp(sz);
        t1.record();
        cublas.gemv('T', sz, q_mat.array().raw_ptr(), sz, sz, real(1), b_x.array().raw_ptr(), real(0), tmp.array().raw_ptr());
        cublas.gemv('N', sz, r_inv_mat.array().raw_ptr(), sz, sz, real(1), tmp.array().raw_ptr(), real(0), b_x.array().raw_ptr());    
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t2.record();
        std::cout << "solve time: " << t2.elapsed_time(t1) << "ms" << std::endl;
    }

    if (solve_algo == 3) 
    {
        //get Q matrix explicitly
        cusolver.orgqr(sz, sz, sz, mat_tmp.array().raw_ptr(), tau.array().raw_ptr(), q_mat.array().raw_ptr());
        //mat_tmp.sync_from_array();
        /// Init r_inv with ident matrix
        scfd::utils::system_timer_event t1_1,t2_1;
        t1_1.record();
        ker_r_inv_matrix_set_ident<<<((sz*sz)/256)+1,256>>>(sz, r_inv_mat.array().raw_ptr());
        vector_t r_diag_inv(sz);
        ker_precalc_invert_r_diag<<<(sz/256)+1,256>>>(sz, mat_tmp.array().raw_ptr(), r_diag_inv.array().raw_ptr());
        ker_invert_r_diag<<<((sz*sz)/256)+1,256>>>(sz, r_diag_inv.array().raw_ptr(), mat_tmp.array().raw_ptr(), r_inv_mat.array().raw_ptr());

        vector_t mat_tmp_col_i(sz);
        for (int i = sz-1;i >= 0;--i)
        {
            ker_copy_mat_tmp_col_i<<<(i/256)+1,256>>>(sz, i, mat_tmp.array().raw_ptr(), mat_tmp_col_i.array().raw_ptr());
            ///sz([i,sz)x[0,i]) = (sz-i)*i
            ker_back_elimination<<<(((sz-i)*i)/256)+1,256>>>(sz, i, mat_tmp_col_i.array().raw_ptr(), r_inv_mat.array().raw_ptr());
            /*std::cout << "i = " << i << std::endl;
            mat_tmp_col_i.sync_from_array();
            for (int ii = 0;ii < i;++ii)
            {
                std::cout << mat_tmp_col_i(ii) << std::endl;
            }*/
            /*r_inv_mat.sync_from_array();
            for (int ii = 0;ii < sz;++ii)
            {
                for (int jj = 0;jj < sz;++jj)
                {
                    std::cout << r_inv_mat(ii,jj) << " ";
                }
                std::cout << std::endl;
            }*/
        }
        t2_1.record();
        std::cout << "invert time: " << t2_1.elapsed_time(t1_1) << "ms" << std::endl;
        vector_t tmp(sz);
        t1.record();
        cublas.gemv('T', sz, q_mat.array().raw_ptr(), sz, sz, real(1), b_x.array().raw_ptr(), real(0), tmp.array().raw_ptr());
        cublas.gemv('N', sz, r_inv_mat.array().raw_ptr(), sz, sz, real(1), tmp.array().raw_ptr(), real(0), b_x.array().raw_ptr());    
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t2.record();
        std::cout << "solve time: " << t2.elapsed_time(t1) << "ms" << std::endl;
    }

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
    int defect, int solve_algo)
{
    matrix_t mat;
    init_array_matrix_with_values(mat, mat_vals);
    vector_t b,x(rhs_vals.size());
    init_array_vector_with_values(b, rhs_vals);

    test_gesv_with_defect_matrix(mat, b, x, defect, solve_algo);
}

void test_gesv_with_defect_matrix_with_1d_poisson(int sz, int solve_algo)
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

    test_gesv_with_defect_matrix(mat, b, x, 1, solve_algo);

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
    int sz = 1000, solve_algo = 0;
    if (argc >= 2)
    {
        sz = std::stoi(args[1]);
    }
    if (argc >= 3)
    {
        solve_algo = std::stoi(args[2]);
    }

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
        4, solve_algo
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
        2, solve_algo  
    );

    std::cout << "test degenerate " << sz << "x" << sz << " poisson 1d matrix with defect=1 and rhs in image" << std::endl;
    test_gesv_with_defect_matrix_with_1d_poisson(sz,solve_algo);

    return 0;
}