#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <scfd/memory/host.h>
#include <scfd/memory/cuda.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>


template<class T, class V>
inline __device__ void calc(int i, int j, int k, V M, int Nx, int Ny, int Nz)
{

    M(i,j,k) = (i+Nx*j+Nx*Ny*k)%256;

}

template<class T, class V>
inline __device__ void calc_ptr(int i, int j, int k, V M, int Nx, int Ny, int Nz)
{

    M[i+Nx*j+Nx*Ny*k] = (i+Nx*j+Nx*Ny*k)%256;

}

template<class T, class V>
__global__ void kernel1(int Nx, int Ny, int Nz, V M)
{
    int idx=blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= Nx*Ny*Nz) return;
    int l = (idx)/(Nx*Ny);
    int jk = idx-l*Nx*Ny;
    int k = (jk)/(Nx);
    int j = jk - k*Nx;
    // if((j>=Nx)||(k>=Ny)||(l>=Nz)) return;

    calc<T,V>(j, k, l, M, Nx, Ny, Nz);
}

template<class T, class V>
__global__ void kernel1_ptr(int Nx, int Ny, int Nz, V M)
{
    int idx=blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= Nx*Ny*Nz) return;
    int l = (idx)/(Nx*Ny);
    int jk = idx-l*Nx*Ny;
    int k = (jk)/(Nx);
    int j = jk - k*Nx;
    // if((j>=Nx)||(k>=Ny)||(l>=Nz)) return;

    calc_ptr<T,V>(j, k, l, M, Nx, Ny, Nz);
}


//#define USE_PTR

int main(int argc, char const *argv[])
{
    using T = double;
    using cuda_mem_t = scfd::memory::cuda_device;
    using host_mem_t = scfd::memory::host;
    using gpu1_t = scfd::arrays::tensor0_array_nd<T, 3, cuda_mem_t>;
    using gpu1_view_t = typename gpu1_t::view_type;

    int Nx = 10, Ny = 20, Nz = 30;
    int NR = Nx*Ny*Nz;

    scfd::utils::init_cuda(-1, -1);

    {
        #ifndef USE_PTR
        gpu1_t ARRAY;
        ARRAY.init(Nx, Ny, Nz);
        #else
        T* PTR = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void**)&PTR, sizeof(T)*NR ));
        #endif

        const unsigned int blocksize = 1024;
        unsigned int nthreads = blocksize;
        size_t k1 = ( NR + nthreads -1 )/nthreads ;
        

        dim3 dimGrid(k1, 1, 1);
        dim3 dimBlock(blocksize, 1, 1);

        #ifndef USE_PTR
        kernel1<T, gpu1_t><<<dimGrid, dimBlock>>>(Nx, Ny, Nz, ARRAY);
        #else
        kernel1_ptr<T, T*><<<dimGrid, dimBlock>>>(Nx, Ny, Nz, PTR);
        CUDA_SAFE_CALL(cudaFree(PTR));
        #endif
    }
    cudaDeviceReset();
    //CUDA_SAFE_CALL(cudaFree(PTR));
    return 0;
}