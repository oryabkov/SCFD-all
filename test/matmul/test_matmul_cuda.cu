#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include <scfd/utils/init_cuda.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/system_timer_event.h>
#include <scfd/utils/cuda_timer_event.h>

#include <scfd/static_vec/vec.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/first_index_fast_arranger.h>
#include <scfd/arrays/last_index_fast_arranger.h>

#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda.h>
#include <scfd/for_each/cuda_impl.cuh>

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp.h>
#include <scfd/for_each/openmp_impl.h>

constexpr std::size_t K  = N_NUM; // matrix size K x K
constexpr std::size_t K2 = K * K;

using timer_event_device_t = scfd::utils::cuda_timer_event;
using timer_event_host_t = scfd::utils::system_timer_event;

using for_each_cuda_t = scfd::for_each::cuda<>;
using mem_device_t = scfd::memory::cuda_device;
using for_each_omp_t = scfd::for_each::openmp<>;
using mem_host_t = scfd::memory::host;


using T = float;
template<scfd::arrays::ordinal_type... Dims>
using gpu_arranger_t = scfd::arrays::first_index_fast_arranger<Dims...>;
template<scfd::arrays::ordinal_type... Dims>
using cpu_arranger_t = scfd::arrays::last_index_fast_arranger<Dims...>;

using array_device_classic_t = scfd::arrays::tensor2_array<T, mem_device_t, K, K>;
using array_device_classic_view_t = array_device_classic_t::view_type;

using array_device_like_host_t = scfd::arrays::tensor2_array<T, mem_device_t, K, K, cpu_arranger_t>;
using array_device_like_host_view_t = array_device_like_host_t::view_type;

//using array_device_t = array_device_like_host_t;
using array_device_t = array_device_classic_t;
using array_device_view_t = array_device_t::view_type;

using array_host_t = scfd::arrays::tensor2_array<T, mem_host_t, K, K>;
using array_host_view_t = array_host_t::view_type;


const int block_size = 128;

#define IC(idx, i, j) (idx)*(K2)+(i)*(K)+(j)
#define IG(idx, i, j) (idx)+(N)*((K)*(j)+(i))


/******************************* BEGIN DEVICE KERNELS ***************************************/
template<class T>
#ifdef USE_CONST
__global__ void mat_mul_kern(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
#else
__global__ void mat_mul_kern(std::size_t N, T* f1_, T* f2_, T* f_out_)
#endif
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;

    for(std::size_t i = 0u; i < K; ++i)
    for(std::size_t j = 0u; j < K; ++j)
    {
        f_out_[IC(idx, i, j)] = 0.f;
        for(std::size_t k = 0u; k < K; ++k)
            f_out_[IC(idx, i, j)] += f1_[IC(idx, i, k)] * f2_[IC(idx, k, j)];
    }
}

template<class T>
#ifdef USE_CONST
__global__ void mat_mul_kern_ok(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
#else
__global__ void mat_mul_kern_ok(std::size_t N, T* f1_, T* f2_, T* f_out_)
#endif
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;

    for(std::size_t i = 0u; i < K; ++i)
    for(std::size_t j = 0u; j < K; ++j)
    {
        f_out_[IG(idx, i, j)] = 0.f;
        for(std::size_t k = 0u; k < K; ++k)
            f_out_[IG(idx, i, j)] += f1_[IG(idx, i, k)] * f2_[IG(idx, k, j)];
    }
}
/******************************* END DEVICE KERNELS ***************************************/



/****************************** BEGIN TENSOR FUNCTOR **************************************/
template<class MatrixND>
struct func_mat_mul
{

    MatrixND f1_;
    MatrixND f2_;
    MatrixND f_out_;

    func_mat_mul(const MatrixND &f1, const MatrixND &f2, MatrixND &f_out):
    f1_(f1),
    f2_(f2),
    f_out_(f_out)
    {}


    __DEVICE_TAG__ void operator()(const int &idx)
    {
        for(std::size_t i = 0u; i < K; ++i)
        for(std::size_t j = 0u; j < K; ++j)
        {
            f_out_(idx, i, j) = 0.f;
            for(std::size_t k = 0u; k < K; ++k)
                f_out_(idx, i, j) += f1_(idx, i, k) * f2_(idx, k, j);
        }
    }
};

template<class ForEach, class MatrixND>
void mat_mul_device_f(const std::size_t N, const MatrixND& u, const MatrixND& v, MatrixND& w)
{

    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_mat_mul<MatrixND>(u, v, w), 0, N);
    for_each.wait();

}
template<class ForEach, class MatrixND>
void mat_mul_host_f(const std::size_t N, const MatrixND& u, const MatrixND& v, MatrixND& w)
{

    ForEach for_each;
    for_each(func_mat_mul<MatrixND>(u, v, w), 0, N);
    for_each.wait();

}
/******************************* END TENSOR FUNCTOR ***************************************/



/******************************** BEGIN PTR FUNCTOR ***************************************/
struct func_mat_mul_ptr
{

    const T* f1_;
    const T* f2_;
    T*    f_out_;
    std::size_t N;

    func_mat_mul_ptr(std::size_t const sz, const T* f1, const T* f2, T* f_out):
    N(sz),
    f1_(f1),
    f2_(f2),
    f_out_(f_out)
    {}


    __DEVICE_TAG__ void operator()(const int &idx)
    {
        for(std::size_t i = 0u; i < K; ++i)
        for(std::size_t j = 0u; j < K; ++j)
        {
            f_out_[IG(idx, i, j)] = 0.f;
            for(std::size_t k = 0u; k < K; ++k)
                f_out_[IG(idx, i, j)] += f1_[IG(idx, i, k)] * f2_[IG(idx, k, j)];
        }
    }
};

template<class ForEach>
void mat_mul_device_ptr(const std::size_t N, const T* u, const T* v, T* w)
{

    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_mat_mul_ptr(N, u, v, w), 0, N);
    for_each.wait();

}
template<class ForEach>
void mat_mul_host_ptr(const std::size_t N, const T* u, const T* v, T* w)
{

    ForEach for_each;
    for_each(func_mat_mul_ptr(N, u, v, w), 0, N);
    for_each.wait();

}
/******************************** END PTR FUNCTOR *****************************************/



template<class T, class MatrixND>
T check_coincide_tensor(const std::size_t N, const T* ptr_, const MatrixND& ta_)
{
    T diff = 0.0;

    for(std::size_t n=0; n<N; ++n)
    for(std::size_t i=0; i<K; ++i)
    for(std::size_t j=0; j<K; ++j)
        diff += std::abs(ptr_[IC(n,i,j)] - ta_(n,i,j) );

    return diff;
}

template<class T>
T check_coincide_ptr(std::size_t N, const T* ptr_, const T* ta_)
{
    T diff = 0.0;

    for(std::size_t n=0; n<N; ++n)
    for(std::size_t i=0; i<K; ++i)
    for(std::size_t j=0; j<K; ++j)
        diff += std::abs(ptr_[IC(n,i,j)] - ta_[IC(n,i,j)] );

    return diff;
}


int main(int argc, char const *argv[])
{

    if(argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " N iters tests" << std::endl;
        std::cout << "where: N is a size of R^{K^2 * N}, iters is number of iterations for better measurements, K is a static parameter." << std::endl;
        std::cout << "       tests = d/h/a for device, host or all." << std::endl;
        return 1;
    }
    std::size_t N = std::atoi(argv[1]);
    std::size_t number_of_iters = std::atoi(argv[2]);
    char tests = argv[3][0];

    std::size_t total_size = K2 * N;

    scfd::utils::init_cuda_persistent();
    int device_id;
    CUDA_SAFE_CALL(cudaGetDevice(&device_id));

    T *u_ptr_host, *v_ptr_host, *mat_mul_ptr_host;
    T *u_ptr_ok_host, *v_ptr_ok_host, *mat_mul_ptr_ok_host;

    T *u_ptr_dev, *v_ptr_dev, *mat_mul_ptr_dev;          //with incorrect GPU layout i.e. CPU layout
    T *u_ptr_ok_dev, *v_ptr_ok_dev, *mat_mul_ptr_ok_dev; //with correct GPU layout
    T *u_ptr_func_dev, *v_ptr_func_dev, *mat_mul_ptr_func_dev; //func with plain ptr

    T *mat_mul_ptr_check, *mat_mul_ptr_ok_check, *mat_mul_ptr_func_check;

    std::random_device rd;
    std::mt19937 engine{ rd() };
    std::uniform_real_distribution<> dist(-100.0, 100.0);


    u_ptr_host                = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_host                = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    u_ptr_ok_host             = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_ok_host             = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_host          = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_check         = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_ok_host       = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_ok_check      = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );


    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_ok_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_ok_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&mat_mul_ptr_ok_dev , sizeof(T)*total_size ) );

    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_dev        , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_dev        , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&mat_mul_ptr_dev  , sizeof(T)*total_size ) );

    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_ok_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_ok_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&mat_mul_ptr_ok_dev , sizeof(T)*total_size ) );

    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_func_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_func_dev       , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&mat_mul_ptr_func_dev , sizeof(T)*total_size ) );

    mat_mul_ptr_func_check    = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );


    array_device_t u_dev, v_dev, mat_mul_dev;
    array_host_t u_host, v_host, mat_mul_host;

    u_dev.init(N); v_dev.init(N); mat_mul_dev.init(N);
    u_host.init(N); v_host.init(N); mat_mul_host.init(N);
    array_device_view_t u_dev_view(u_dev), v_dev_view(v_dev), mat_mul_dev_view(mat_mul_dev);
    array_host_t u_host_view(u_host), v_host_view(v_host), mat_mul_host_view(mat_mul_host);

    timer_event_host_t host_e1, host_e2;
    timer_event_device_t device_e1, device_e2;
    timer_event_device_t device_int_e1, device_int_e2;

    #pragma omp parallel for
    for(std::size_t n=0u; n<N; ++n)
    for(std::size_t i=0u; i<K; ++i)
    for(std::size_t j=0u; j<K; ++j)
    {
        T u_ = dist(engine);
        T v_ = dist(engine);

        u_ptr_host[IC(n,i,j)] = u_;
        v_ptr_host[IC(n,i,j)] = v_;

        u_dev_view(n,i,j) = u_;
        v_dev_view(n,i,j) = v_;

        mat_mul_dev_view(n,i,j) = 0.0;
        u_host_view(n,i,j)    = u_;
        v_host_view(n,i,j)    = v_;

        mat_mul_host_view(n,i,j) = 0.0;
        u_ptr_ok_host[IG(n,i,j)] = u_;
        v_ptr_ok_host[IG(n,i,j)] = v_;
    }

    for(std::size_t n=0u; n<N; ++n)
    for(std::size_t i=0u; i<K; ++i)
    for(std::size_t j=0u; j<K; ++j)
    {
        mat_mul_ptr_host[IC(n,i,j)] = 0.0;
        for(std::size_t k=0u; k < K; ++k)
            mat_mul_ptr_host[IC(n,i,j)] += u_ptr_host[IC(n,i,k)] * v_ptr_host[IC(n,k,j)];

        mat_mul_ptr_ok_host[IG(n,i,j)] = 0.0;
        for(std::size_t k=0u; k < K; ++k)
            mat_mul_ptr_ok_host[IG(n,i,j)] += u_ptr_ok_host[IG(n,i,k)] * v_ptr_ok_host[IG(n,k,j)];
    }

    CUDA_SAFE_CALL( cudaMemcpy( (void*) u_ptr_func_dev,   (void*)u_ptr_ok_host,      sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) v_ptr_func_dev,   (void*)v_ptr_ok_host,      sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) u_ptr_dev,        (void*)u_ptr_host,         sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) v_ptr_dev,        (void*)v_ptr_host,         sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) u_ptr_ok_dev,     (void*)u_ptr_ok_host,      sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) v_ptr_ok_dev,     (void*)v_ptr_ok_host,      sizeof(T)*total_size, cudaMemcpyHostToDevice ) );

    u_dev_view.release(true);
    v_dev_view.release(true);
    mat_mul_dev_view.release(true);

    if((tests == 'd')||(tests == 'a'))
    {
        std::vector<double> gpu_tensor; gpu_tensor.reserve(number_of_iters);

        //WARM UP
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            mat_mul_device_f<for_each_cuda_t, array_device_t>(N, u_dev, v_dev, mat_mul_dev);
        }

        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_device_f<for_each_cuda_t, array_device_t>(N, u_dev, v_dev, mat_mul_dev);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_tensor.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
        std::cout << "device tensor time       = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        /***********************************************************************************************************/

        std::vector<double> gpu_ptr_func; gpu_ptr_func.reserve(number_of_iters);

        //WARM UP
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            mat_mul_device_ptr<for_each_cuda_t>(N, u_ptr_func_dev, v_ptr_func_dev, mat_mul_ptr_func_dev);
        }

        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_device_ptr<for_each_cuda_t>(N, u_ptr_func_dev, v_ptr_func_dev, mat_mul_ptr_func_dev);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr_func.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
        std::cout << "device ptr_func time     = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        /**************************************************************************************************************/

        dim3 dimBlock(block_size,1);
        dim3 dimGrid( (N/block_size)+1,1);

        std::vector<double> gpu_ptr; gpu_ptr.reserve(number_of_iters);

        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_kern<T><<<dimGrid, dimBlock>>>(N, u_ptr_dev, v_ptr_dev, mat_mul_ptr_dev);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr.push_back( elapsed_seconds.count() );
        }
        device_e2.record();
#ifdef USE_CONST
        std::cout << "device ptr_const time    = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#else
        std::cout << "device ptr time          = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#endif
        /***************************************************************************************************************/

        std::vector<double> gpu_ptr_ok; gpu_ptr_ok.reserve(number_of_iters);
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_kern_ok<T><<<dimGrid, dimBlock>>>(N, u_ptr_ok_dev, v_ptr_ok_dev, mat_mul_ptr_ok_dev);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr_ok.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
#ifdef USE_CONST
        std::cout << "device ptr_ok_const time = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#else
        std::cout << "device ptr_ok time       = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#endif
        /***************************************************************************************************************/

        mat_mul_dev_view.init(mat_mul_dev, true);
        std::cout << "gpu tensor diff          = " << check_coincide_tensor(N, mat_mul_ptr_host, mat_mul_dev_view) << std::endl;
        mat_mul_dev_view.release(false);

        CUDA_SAFE_CALL( cudaMemcpy( (void*) mat_mul_ptr_func_check, (void*)mat_mul_ptr_func_dev, sizeof(T)*total_size, cudaMemcpyDeviceToHost ) );
        std::cout << "gpu ptr_func diff        = " << check_coincide_ptr(N, mat_mul_ptr_ok_host,  mat_mul_ptr_func_check) << std::endl;


        CUDA_SAFE_CALL( cudaMemcpy( (void*) mat_mul_ptr_check, (void*)mat_mul_ptr_dev, sizeof(T)*total_size, cudaMemcpyDeviceToHost ) );
#ifdef USE_CONST
        std::cout << "gpu ptr_const diff       = " << check_coincide_ptr(N, mat_mul_ptr_host, mat_mul_ptr_check) << std::endl;
#else
        std::cout << "gpu ptr    diff          = " << check_coincide_ptr(N, mat_mul_ptr_host, mat_mul_ptr_check) << std::endl;
#endif

        CUDA_SAFE_CALL( cudaMemcpy( (void*) mat_mul_ptr_ok_check, (void*)mat_mul_ptr_ok_dev, sizeof(T)*total_size, cudaMemcpyDeviceToHost ) );
#ifdef USE_CONST
        std::cout << "gpu ptr_ok_const diff    = " << check_coincide_ptr(N, mat_mul_ptr_ok_host, mat_mul_ptr_ok_check) << std::endl;
#else
        std::cout << "gpu ptr_ok diff          = " << check_coincide_ptr(N, mat_mul_ptr_ok_host, mat_mul_ptr_ok_check) << std::endl;
#endif
        /***************************************************************************************************************/

        std::string filename;
        filename = "execution_times_array_3d_mat_mul_product_" + std::to_string(device_id) + ".csv";
        std::fstream out_file{filename, out_file.out};
        if (!out_file.is_open())
            std::cout << "failed to open " << filename << '\n';
        else
        {
            out_file << "tensor,ptr_func,ptr_bad,ptr_ok" << std::endl;
            for(int j = 0; j<number_of_iters; j++)
            {
                out_file << gpu_tensor.at(j) << ", " << gpu_ptr_func.at(j) << "," << gpu_ptr.at(j) << "," << gpu_ptr_ok.at(j) << std::endl;
            }
            out_file.close();
        }

    }
    if((tests == 'h')||(tests == 'a'))
    {


        std::cout << "executing host tests ... " << std::endl;
        std::vector<double> host_tensor; host_tensor.reserve(number_of_iters);
        std::vector<double> host_ptr; host_ptr.reserve(number_of_iters);
        std::vector<double> host_ptr_ok; host_ptr_ok.reserve(number_of_iters);

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_host_f<for_each_omp_t, array_host_t>(N, u_host, v_host, mat_mul_host);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_tensor.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host tensor time =  " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /**********************************************************************************************************/

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for(std::size_t n=0; n<N; ++n)
            {
                for(std::size_t i=0u; i<K; ++i)
                for(std::size_t j=0u; j<K; ++j)
                {
                    mat_mul_ptr_ok_host[IG(n,i,j)] = 0.f;
                    for(std::size_t k=0u; k<K; ++k)
                        mat_mul_ptr_ok_host[IG(n,i,j)] += u_ptr_ok_host[IG(n,i,k)] * v_ptr_ok_host[IG(n,k,j)];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr_ok.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host ptr_ok time =  " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /*********************************************************************************************************/

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for(std::size_t n=0; n<N; ++n)
            {
                for(std::size_t i=0u; i<K; ++i)
                for(std::size_t j=0u; j<K; ++j)
                {
                    mat_mul_ptr_ok_host[IG(n,i,j)] = 0.f;
                    for(std::size_t k=0u; k<K; ++k)
                        mat_mul_ptr_host[IC(n,i,j)] += u_ptr_host[IC(n,i,k)] * v_ptr_host[IC(n,k,j)];
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host ptr time =" <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /*******************************************************************************************************/

        std::string filename;
        filename = "execution_times_array_3d_mat_mul_product_host.csv";
        std::fstream out_file_cpu{filename, out_file_cpu.out};
        if (!out_file_cpu.is_open())
            std::cout << "failed to open " << filename << '\n';
        else
        {
            out_file_cpu << "tensor,ptr_bad,ptr_ok" << std::endl;
            for(int j = 0; j<number_of_iters; j++)
            {
                out_file_cpu << host_tensor.at(j) << "," << host_ptr.at(j) << "," << host_ptr_ok.at(j) << std::endl;
            }
            out_file_cpu.close();
        }

    }

    std::free(u_ptr_ok_host);
    std::free(v_ptr_ok_host);

    std::free(mat_mul_ptr_host);
    std::free(mat_mul_ptr_ok_host);

    std::free(u_ptr_host);
    std::free(v_ptr_host);

    std::free(mat_mul_ptr_check);
    std::free(mat_mul_ptr_ok_check);
    std::free(mat_mul_ptr_func_check);

    CUDA_SAFE_CALL( cudaFree(mat_mul_ptr_ok_dev) );
    CUDA_SAFE_CALL( cudaFree(v_ptr_ok_dev) );
    CUDA_SAFE_CALL( cudaFree(u_ptr_ok_dev) );

    CUDA_SAFE_CALL( cudaFree(mat_mul_ptr_dev) );
    CUDA_SAFE_CALL( cudaFree(v_ptr_dev) );
    CUDA_SAFE_CALL( cudaFree(u_ptr_dev) );

    CUDA_SAFE_CALL( cudaFree(mat_mul_ptr_func_dev) );
    CUDA_SAFE_CALL( cudaFree(v_ptr_func_dev) );
    CUDA_SAFE_CALL( cudaFree(u_ptr_func_dev) );

    return 0;
}
