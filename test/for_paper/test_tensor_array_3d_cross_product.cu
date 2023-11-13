#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT

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

using array_device_classic_t = scfd::arrays::tensor1_array<T, mem_device_t, 3>;
using array_device_classic_view_t = array_device_classic_t::view_type;

using array_device_like_host_t = scfd::arrays::tensor1_array<T, mem_device_t, 3, cpu_arranger_t>;
using array_device_like_host_view_t = array_device_like_host_t::view_type;

using array_device_t = array_device_like_host_t;
using array_device_view_t = array_device_like_host_view_t;


using array_host_t = scfd::arrays::tensor1_array<T, mem_host_t, 3>;
using array_host_view_t = array_host_t::view_type;

using array3_device_t = scfd::arrays::tensor0_array_nd<T, 3, mem_device_t>;


const int block_size = 128;

#define IC(j,k) (3)*(j)+(k)
#define IG(j,k) (j)+(N)*(k)


//corss product using kernel
// template<class Vec3>
// __global__ void cross_prod_kern(std::size_t N, const Vec3 f1_, const Vec3 f2_, Vec3 f_out_)
// {
//     int     idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx>=N) return;

//     f_out_(idx,0) = f1_(idx,1)*f2_(idx,2)-f1_(idx,2)*f2_(idx,1);
//     f_out_(idx,1) = -( f1_(idx,0)*f2_(idx,2)-f1_(idx,2)*f2_(idx,0) );
//     f_out_(idx,2) = f1_(idx,0)*f2_(idx,1)-f1_(idx,1)*f2_(idx,0);

// }

template<class T>
__global__ void cross_prod_kern(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
{
    int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;
    f_out_[IC(idx, 0)] = f1_[IC(idx,1)]*f2_[IC(idx,2)]-f1_[IC(idx,2)]*f2_[IC(idx,1)];
    f_out_[IC(idx, 1)] = -( f1_[IC(idx,0)]*f2_[IC(idx,2)]-f1_[IC(idx,2)]*f2_[IC(idx,0)] );
    f_out_[IC(idx, 2)] = f1_[IC(idx,0)]*f2_[IC(idx,1)]-f1_[IC(idx,1)]*f2_[IC(idx,0)];
}

template<class T>
__global__ void cross_prod_kern_ok(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
{
    int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;
    f_out_[IG(idx, 0)] = f1_[IG(idx,1)]*f2_[IG(idx,2)]-f1_[IG(idx,2)]*f2_[IG(idx,1)];
    f_out_[IG(idx, 1)] = -( f1_[IG(idx,0)]*f2_[IG(idx,2)]-f1_[IG(idx,2)]*f2_[IG(idx,0)] );
    f_out_[IG(idx, 2)] = f1_[IG(idx,0)]*f2_[IG(idx,1)]-f1_[IG(idx,1)]*f2_[IG(idx,0)];
}


// cross product using foreach
template<class Vec3>
struct func_cross_prod
{

    Vec3 f1_;
    Vec3 f2_;
    Vec3 f_out_;
    
    func_cross_prod(const Vec3 &f1,const Vec3 &f2, Vec3 &f_out): 
    f1_(f1), 
    f2_(f2), 
    f_out_(f_out)
    {}
    

    __DEVICE_TAG__ void operator()(const int &idx)
    {
        f_out_(idx,0) = f1_(idx,1)*f2_(idx,2)-f1_(idx,2)*f2_(idx,1);
        f_out_(idx,1) = -( f1_(idx,0)*f2_(idx,2)-f1_(idx,2)*f2_(idx,0) );
        f_out_(idx,2) = f1_(idx,0)*f2_(idx,1)-f1_(idx,1)*f2_(idx,0);

    }
};

template<class ForEach, class Vec3>
void cross_prod_device(const std::size_t N, const Vec3& u, const Vec3& v, Vec3& w)
{
    
    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_cross_prod<Vec3>(u, v, w), 0, N);
    for_each.wait();

}
template<class ForEach, class Vec3>
void cross_prod_host(const std::size_t N, const Vec3& u, const Vec3& v, Vec3& w)
{
    
    ForEach for_each;
    for_each(func_cross_prod<Vec3>(u, v, w), 0, N);
    for_each.wait();

}


template<class T, class Vec3>
T check_coincide_tensor(const std::size_t N, const T* ptr_, const Vec3& ta_)
{
    T diff = 0.0;
    for(std::size_t j=0; j<N; j++ )
    {
        for(std::size_t k=0; k<3; k++)
        {
            diff += std::abs(ptr_[3*j+k] - ta_(j,k) );
        }
    }
    return diff;
}

template<class T>
T check_coincide_ptr(std::size_t N, const T* ptr_, const T* ta_)
{
    T diff = 0.0;
    for(std::size_t j=0; j<N; j++ )
    {
        for(std::size_t k=0; k<3; k++)
        {
            diff += std::abs(ptr_[3*j+k] - ta_[3*j+k] );
        }
    }
    return diff;
}

int main(int argc, char const *argv[])
{

    if(argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " N iters tests" << std::endl;
        std::cout << "where: N is a size of R^{3XN}, iters is number of iterations for better measurements," << std::endl;
        std::cout << "       tests = d/h/a for device, host or all." << std::endl;
        return 1;
    }
    std::size_t N = std::atoi(argv[1]);
    std::size_t number_of_iters = std::atoi(argv[2]);
    char tests = argv[3][0];

    std::size_t total_size = 3*N;

    scfd::utils::init_cuda_persistent();
    //scfd::utils::init_cuda(-1, -1);
    int device_id;
    CUDA_SAFE_CALL(cudaGetDevice(&device_id));

    T *u_ptr_host, *v_ptr_host, *cross_ptr_host, *cross_ptr_host_check;
    T *u_ptr_ok_host, *v_ptr_ok_host, *cross_ptr_ok_host, *cross_ptr_ok_host_check;
    T *u_ptr_dev, *v_ptr_dev, *cross_ptr_dev; //with incorrect GPU layout i.e. CPU layout
    T *u_ptr_ok_dev, *v_ptr_ok_dev, *cross_ptr_ok_dev; //with correct GPU layout


    std::random_device rd;
    std::mt19937 engine{ rd() }; 
    std::uniform_real_distribution<> dist(-100.0, 100.0);

    // auto gen_rand = [&dist, &engine]()
    // {
    //     return dist(engine);
    // };

    //TODO: add throw
    u_ptr_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    cross_ptr_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    cross_ptr_host_check = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    u_ptr_ok_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_ok_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    cross_ptr_ok_host = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    cross_ptr_ok_host_check = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );    
    //
    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_dev , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_dev , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&cross_ptr_dev , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&u_ptr_ok_dev , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&v_ptr_ok_dev , sizeof(T)*total_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&cross_ptr_ok_dev , sizeof(T)*total_size ) );

    
    array_device_t u_dev, v_dev, cross_dev;
    array_host_t u_host, v_host, cross_host;
    
    u_dev.init(N); v_dev.init(N); cross_dev.init(N);
    u_host.init(N); v_host.init(N); cross_host.init(N);
    array_device_view_t u_dev_view(u_dev), v_dev_view(v_dev), cross_dev_view(cross_dev);
    array_host_t u_host_view(u_host), v_host_view(v_host), cross_host_view(cross_host);

    timer_event_host_t host_e1, host_e2;
    timer_event_device_t device_e1, device_e2;
    timer_event_device_t device_int_e1, device_int_e2;

    #pragma omp parallel for
    for(std::size_t j=0; j<N; j++ )
    {
        for(std::size_t k=0; k<3; k++)
        {
            T u_ = dist(engine);
            T v_ = dist(engine);
            u_ptr_host[IC(j,k)] = u_;
            v_ptr_host[IC(j,k)] = v_;
            u_dev_view(j,k) = u_;
            v_dev_view(j,k) = v_;
            cross_dev_view(j,k) = 0.0;
            u_host_view(j,k) = u_;
            v_host_view(j,k) = v_;
            cross_host_view(j,k) = 0.0;
            u_ptr_ok_host[IG(j,k)] = u_;
            v_ptr_ok_host[IG(j,k)] = v_;
        }
        cross_ptr_host[IC(j,0)] = u_ptr_host[IC(j,1)]*v_ptr_host[IC(j,2)] - u_ptr_host[IC(j,2)]*v_ptr_host[IC(j,1)];
        cross_ptr_host[IC(j,1)] = -(u_ptr_host[IC(j,0)]*v_ptr_host[IC(j,2)] - u_ptr_host[IC(j,2)]*v_ptr_host[IC(j,0)]);
        cross_ptr_host[IC(j,2)] = u_ptr_host[IC(j,0)]*v_ptr_host[IC(j,1)] - u_ptr_host[IC(j,1)]*v_ptr_host[IC(j,0)];
        cross_ptr_ok_host[IG(j,0)] = u_ptr_ok_host[IG(j,1)]*v_ptr_ok_host[IG(j,2)] - u_ptr_ok_host[IG(j,2)]*v_ptr_ok_host[IG(j,1)];
        cross_ptr_ok_host[IG(j,1)] = -(u_ptr_ok_host[IG(j,0)]*v_ptr_ok_host[IG(j,2)] - u_ptr_ok_host[IG(j,2)]*v_ptr_ok_host[IG(j,0)]);
        cross_ptr_ok_host[IG(j,2)] = u_ptr_ok_host[IG(j,0)]*v_ptr_ok_host[IG(j,1)] - u_ptr_ok_host[IG(j,1)]*v_ptr_ok_host[IG(j,0)];        
    }
    CUDA_SAFE_CALL( cudaMemcpy( (void*) u_ptr_dev, (void*)u_ptr_host, sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) v_ptr_dev, (void*)v_ptr_host, sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) u_ptr_ok_dev, (void*)u_ptr_ok_host, sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) v_ptr_ok_dev, (void*)v_ptr_ok_host, sizeof(T)*total_size, cudaMemcpyHostToDevice ) );
    u_dev_view.release(true); v_dev_view.release(true); cross_dev_view.release(true);

    if((tests == 'd')||(tests == 'a'))
    {
        //WARM UP
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {   
            cross_prod_device<for_each_cuda_t, array_device_t>(N, u_dev, v_dev, cross_dev);
        }

        std::vector<double> gpu_tensor; gpu_tensor.reserve(number_of_iters);
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {   
            auto start = std::chrono::high_resolution_clock::now();
            cross_prod_device<for_each_cuda_t, array_device_t>(N, u_dev, v_dev, cross_dev);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_tensor.push_back( elapsed_seconds.count() );
        }
        device_e2.record();
        std::cout << "device tensor time = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        dim3 dimBlock(block_size,1);
        dim3 dimGrid( (N/block_size)+1,1);

        std::vector<double> gpu_ptr; gpu_ptr.reserve(number_of_iters);

        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cross_prod_kern<T><<<dimGrid, dimBlock>>>(N, u_ptr_dev, v_ptr_dev, cross_ptr_dev);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr.push_back( elapsed_seconds.count() );       
        }
        device_e2.record();
        std::cout << "device ptr time =    " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
        
        std::vector<double> gpu_ptr_ok; gpu_ptr_ok.reserve(number_of_iters);
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            cross_prod_kern_ok<T><<<dimGrid, dimBlock>>>(N, u_ptr_ok_dev, v_ptr_ok_dev, cross_ptr_ok_dev);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr_ok.push_back( elapsed_seconds.count() ); 
        }
        device_e2.record();
        std::cout << "device ptr_ok time =  " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        cross_dev_view.init(cross_dev, true);
        std::cout << "gpu tensor diff = " << check_coincide_tensor(N, cross_ptr_host, cross_dev_view) << std::endl;
        cross_dev_view.release(false);

        CUDA_SAFE_CALL( cudaMemcpy( (void*) cross_ptr_host_check, (void*)cross_ptr_dev, sizeof(T)*total_size, cudaMemcpyDeviceToHost ) );
        std::cout << "gpu ptr diff    = " << check_coincide_ptr(N, cross_ptr_host, cross_ptr_host_check) << std::endl;
        CUDA_SAFE_CALL( cudaMemcpy( (void*) cross_ptr_ok_host_check, (void*)cross_ptr_ok_dev, sizeof(T)*total_size, cudaMemcpyDeviceToHost ) );
        std::cout << "gpu ptr diff    = " << check_coincide_ptr(N, cross_ptr_ok_host, cross_ptr_ok_host_check) << std::endl;

        std::string filename;
        filename = "executtion_times_array_3d_cross_product_" + std::to_string(device_id) + ".csv";
        std::fstream out_file{filename, out_file.out};
        if (!out_file.is_open())
            std::cout << "failed to open " << filename << '\n';
        else
        {
            out_file << "tensor,ptr_bad,ptr_ok" << std::endl;
            for(int j = 0; j<number_of_iters; j++)
            {
                out_file << gpu_tensor.at(j) << "," << gpu_ptr.at(j) << "," << gpu_ptr_ok.at(j) << std::endl;
            }
            out_file.close();
        }

    }
    else if((tests == 'h')||(tests == 'a'))
    {

    
        std::cout << "executing host tests ... " << std::endl;
        std::vector<double> host_tensor; host_tensor.reserve(number_of_iters);
        std::vector<double> host_ptr; host_ptr.reserve(number_of_iters);
        std::vector<double> host_ptr_ok; host_ptr_ok.reserve(number_of_iters);        

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {   
            auto start = std::chrono::high_resolution_clock::now();
            cross_prod_host<for_each_omp_t, array_host_t>(N, u_host, v_host, cross_host);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_tensor.push_back( elapsed_seconds.count() );   

        }
        host_e2.record();
        std::cout << "host tensor time =  " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {   
            auto start = std::chrono::high_resolution_clock::now();
            
            #pragma omp parallel for
            for(std::size_t j=0; j<N; j++ )
            {
                cross_ptr_ok_host[IG(j,0)] = u_ptr_ok_host[IG(j,1)]*v_ptr_ok_host[IG(j,2)] - u_ptr_ok_host[IG(j,2)]*v_ptr_ok_host[IG(j,1)];
                cross_ptr_ok_host[IG(j,1)] = -(u_ptr_ok_host[IG(j,0)]*v_ptr_ok_host[IG(j,2)] - u_ptr_ok_host[IG(j,2)]*v_ptr_ok_host[IG(j,0)]);
                cross_ptr_ok_host[IG(j,2)] = u_ptr_ok_host[IG(j,0)]*v_ptr_ok_host[IG(j,1)] - u_ptr_ok_host[IG(j,1)]*v_ptr_ok_host[IG(j,0)]; 
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr_ok.push_back( elapsed_seconds.count() ); 

        }
        host_e2.record();
        std::cout << "host ptr_ok time =  " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;


        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {   
            auto start = std::chrono::high_resolution_clock::now();
            
            #pragma omp parallel for
            for(std::size_t j=0; j<N; j++ )
            {
                cross_ptr_host[IC(j,0)] = u_ptr_host[IC(j,1)]*v_ptr_host[IC(j,2)] - u_ptr_host[IC(j,2)]*v_ptr_host[IC(j,1)];
                cross_ptr_host[IC(j,1)] = -(u_ptr_host[IC(j,0)]*v_ptr_host[IC(j,2)] - u_ptr_host[IC(j,2)]*v_ptr_host[IC(j,0)]);
                cross_ptr_host[IC(j,2)] = u_ptr_host[IC(j,0)]*v_ptr_host[IC(j,1)] - u_ptr_host[IC(j,1)]*v_ptr_host[IC(j,0)];           
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr.push_back( elapsed_seconds.count() ); 

        }
        host_e2.record();
        std::cout << "host ptr time =" <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;





        std::string filename;
        filename = "execution_times_array_3d_cross_product_host.csv";
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


    CUDA_SAFE_CALL( cudaFree(cross_ptr_ok_dev) );
    CUDA_SAFE_CALL( cudaFree(v_ptr_ok_dev) );
    CUDA_SAFE_CALL( cudaFree(u_ptr_ok_dev) );
    std::free(u_ptr_ok_host);
    std::free(v_ptr_ok_host);
    std::free(cross_ptr_ok_host);
    std::free(cross_ptr_ok_host_check);

    CUDA_SAFE_CALL( cudaFree(cross_ptr_dev) );
    CUDA_SAFE_CALL( cudaFree(v_ptr_dev) );
    CUDA_SAFE_CALL( cudaFree(u_ptr_dev) );
    std::free(u_ptr_host);
    std::free(v_ptr_host);
    std::free(cross_ptr_host);
    std::free(cross_ptr_host_check);





    return 0;
}
