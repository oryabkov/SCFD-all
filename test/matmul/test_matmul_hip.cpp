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

#include <scfd/utils/init_hip.h>
#include <scfd/utils/hip_timer_event.h>
#include <scfd/utils/hip_safe_call.h>
#include <scfd/utils/system_timer_event.h>
#include <scfd/utils/hip_timer_event.h>

#include <scfd/static_vec/vec.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/first_index_fast_arranger.h>
#include <scfd/arrays/last_index_fast_arranger.h>

#include <scfd/memory/hip.h>
#include <scfd/for_each/hip.h>
#include <scfd/for_each/hip_impl.h>

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp.h>
#include <scfd/for_each/openmp_impl.h>

constexpr std::size_t K  = N_NUM; // matrix size K x K
constexpr std::size_t K2 = K * K;

using timer_event_device_t = scfd::utils::hip_timer_event;
using timer_event_host_t = scfd::utils::system_timer_event;

using for_each_device_t  = scfd::for_each::hip<>;
using mem_device_t = scfd::memory::hip_device;
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

#include "all_kernels.h"

int main(int argc, char const *argv[])
{

    #define __COMMON_PARTS_DEVICE_INIT__ scfd::utils::init_hip_persistent();
    #define __COMMON_PARTS_SAFE_CALL__  HIP_SAFE_CALL
    #define __COMMON_PARTS_DEVICE_MALLOC__ hipMalloc
    #define __COMMON_PARTS_DEVICE_MEMCPY__ hipMemcpy
    #define __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ hipMemcpyHostToDevice
    #define __COMMON_PARTS_DEVICE_MEMCPY_DEVICE_TO_HOST__ hipMemcpyDeviceToHost
    #define __COMMON_PARTS_DEVICE_SYNCRONIZE__ hipDeviceSynchronize
    #define __COMMON_PARTS_DEVICE_FREE__ hipFree

    #include "common_parts.h"

    return 0;
}
