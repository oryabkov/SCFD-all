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


constexpr std::size_t K  = N_NUM; // matrix size K x K
constexpr std::size_t K2 = K * K;



using timer_event_host_t = scfd::utils::system_timer_event;
using timer_event_device_t = timer_event_host_t;

using for_each_device_t  = scfd::for_each::sycl<>;
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

int main(int argc, char const *argv[])
{
	

	return 0;
}