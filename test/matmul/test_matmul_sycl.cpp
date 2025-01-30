#include "config_scfd.h"

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

#include <scfd/utils/init_sycl.h>
#include <scfd/utils/timer_event.h>
#include <scfd/utils/system_timer_event.h>

#include <scfd/static_vec/vec.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/first_index_fast_arranger.h>
#include <scfd/arrays/last_index_fast_arranger.h>

#include <scfd/memory/sycl.h>
#include <scfd/for_each/sycl.h>
#include <scfd/for_each/sycl_impl.h>

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp.h>
#include <scfd/for_each/openmp_impl.h>

#include <syclcompat.hpp>


constexpr std::size_t K  = N_NUM; // matrix size K x K
constexpr std::size_t K2 = K * K;



using timer_event_host_t = scfd::utils::system_timer_event;
using timer_event_device_t = timer_event_host_t;

using for_each_device_t  = scfd::for_each::sycl_<>;
using mem_device_t = scfd::memory::sycl_device;
using for_each_omp_t = scfd::for_each::openmp<>;
using mem_host_t = scfd::memory::host;


using T = REAL;
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

// #include "all_kernels.h"

struct sycl_selector
{   
    static int cnt;
    static std::size_t free_mem;
    std::size_t max_free_mem = 0;
    bool mem_set;
    sycl_selector():
    mem_set(false)
    {}
    sycl_selector(std::size_t free_mem):
    max_free_mem(free_mem),
    mem_set(true)
    {}

    int operator ()(const sycl::device& device)
    {
        
        if (!mem_set) std::cout << "cnt: " << cnt << "  Device: " << device.get_info<sycl::info::device::name>();
        auto device_info = device.get_info<sycl::info::device::device_type>();
        if ( device_info == sycl::info::device_type::gpu )//device.is_gpu() )//get_info<sycl::device::is_gpu()>() )
        {
            std::size_t total_mem_, free_mem_;
            syclcompat::device_ext de(device);
            de.get_memory_info(free_mem_, total_mem_);
            if( max_free_mem == 0)
            {
                if (free_mem < free_mem_ )
                {
                    free_mem = free_mem_;
                }
            }
            if(!mem_set) std::cout << " is a gpu, total memory: " << total_mem_ << " bytes, free memory: " << free_mem_ << " bytes." << std::endl;
        }
        if( device_info == sycl::info::device_type::cpu )
        {
            if(!mem_set) std::cout << " is a cpu " << std::endl;
        }
        if( device_info == sycl::info::device_type::accelerator)
        {
            if(!mem_set) std::cout << " is an accelerator " << std::endl;       
        }
        if( device_info == sycl::info::device_type::custom)
        {
            if(!mem_set) std::cout << " is a custom device" << std::endl;
        }

        // if( device.get_info<sycl::info::device::name>() )
        // {
        //  std::cout << " is a custom device " << std::endl;
        // }        
        if(!mem_set)
            return 0;
        else
        {
            // cnt++;
            // if(cnt == 2)
            //     return  100;
            // else
            //     return 0;
            if (free_mem == max_free_mem)
            {
                return 100;
            }
            else
            {
                return 0;
            }
        }
    }
};

int sycl_selector::cnt = 0;
std::size_t sycl_selector::free_mem = 0;

int main(int argc, char const *argv[])
{
  
  // for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {

  //   std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
  //   // std::cout << "  Device: " << device.get_info<sycl::info::device::parent_device>() << std::endl;

  // }

    {
        sycl_selector ss;
        sycl::device my_gpu { ss };
    }
    sycl_selector ss(sycl_selector::free_mem);
    sycl::device my_gpu { ss };

    std::cout << "current_device = " << my_gpu.get_info<sycl::info::device::name>() << std::endl;

    // sycl::device my_gpu { sycl::gpu_selector_v };
    // std::cout << "selected device: " << sycl_device_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    // std::cout << "global_mem_size: " <<sycl_device_queue.get_device().get_info<sycl::info::device::global_mem_size>() << " bytes." << std::endl;



    // #define __COMMON_PARTS_DEVICE_INIT__ std::cout << "Selected device: " <<  sycl_device_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    // #define __COMMON_PARTS_SAFE_CALL__  
    // #define __COMMON_PARTS_DEVICE_MALLOC__ hipMalloc
    // #define __COMMON_PARTS_DEVICE_MEMCPY__ hipMemcpy
    // #define __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ hipMemcpyHostToDevice
    // #define __COMMON_PARTS_DEVICE_MEMCPY_DEVICE_TO_HOST__ hipMemcpyDeviceToHost
    // #define __COMMON_PARTS_DEVICE_SYNCRONIZE__ hipDeviceSynchronize
    // #define __COMMON_PARTS_DEVICE_FREE__ hipFree
    // #define __COMMON_PARTS_MEM_GET_INFO__ hipMemGetInfo


    // #include "common_parts.h"

    return 0;
}