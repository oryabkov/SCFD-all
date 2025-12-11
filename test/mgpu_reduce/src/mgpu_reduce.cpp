// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#define SCFD_ARRAYS_ORDINAL_TYPE long int


#if defined (CUDA)
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#elif defined (HIP)
#include <scfd/utils/init_hip_mpi.h>
#include <scfd/utils/hip_safe_call.h>
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd.h>
#include <scfd/for_each/hip_nd_impl.h>
#endif


#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/utils/log_mpi.h>

// #define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>

#include <thrust/device_ptr.h>

#include <vector_operations.h>
#include <functors.h>
#include <vector>
#include <numeric>
#include <scfd/utils/mpi_timer_event.h>
#include <quick_test_check.h>



template<class BaseT>
std::vector<std::size_t> get_max_size_per_gpu(scfd::communication::mpi_comm_info mpi)
{
    std::size_t free_mem_l;
    std::size_t total_mem_l;
    #if defined(CUDA)
        CUDA_SAFE_CALL( cudaMemGetInfo ( &free_mem_l, &total_mem_l ) );
    #elif defined(HIP)
        HIP_SAFE_CALL( hipMemGetInfo ( &free_mem_l, &total_mem_l ) );
    #endif
    auto max_array_size = free_mem_l/sizeof(BaseT);
    std::vector<std::size_t> r(mpi.num_procs, 0);
    mpi.all_gather<std::size_t>(&max_array_size, 1, r.data(), 1);
    mpi.barrier();
    return r;
}



int main(int argc, char *argv[])
{
    static const int dim = 3;
    using log_t = scfd::utils::log_mpi;
    using value_t = double;
    using big_ordinal = long int;
    #if defined(CUDA)
    using mem_t = scfd::memory::cuda_device;
    using for_each_t = scfd::for_each::cuda_nd<dim, big_ordinal>;
    #elif defined(HIP)
    using mem_t = scfd::memory::hip_device;
    using for_each_t = scfd::for_each::hip_nd<dim, big_ordinal>;    
    #endif

    using array_t = scfd::arrays::array_nd<value_t, dim, mem_t>;
    using array_view_t = typename array_t::view_type;
    using vec_ops_t = vector_operations<value_t, array_t>; 
    using vec_t = scfd::static_vec::vec<big_ordinal, dim>;
    using rect_t = scfd::static_vec::rect<big_ordinal, dim>;
    using timer_t = scfd::utils::mpi_timer_event;
    static const double mem_factor = 0.98; //reduce array size due to thrust::reduce bug

    scfd::communication::mpi_wrap mpi(argc, argv);

    int number_of_reps = 1;
    if(argc == 2)
    {
        number_of_reps = std::stoi(argv[1]);
    }

    int myid = mpi.comm_world().myid;
    int nproc = mpi.comm_world().num_procs;
    log_t log;
    timer_t t1,t2;
    t1.record();
    #if defined (CUDA)
    auto device_id = scfd::utils::init_cuda_mpi( mpi.comm_world());
    #elif defined (HIP)
    auto device_id = scfd::utils::init_hip_mpi( mpi.comm_world());
    #endif

    vec_ops_t vec_ops( &mpi.data );
    for_each_t for_each;
    auto sizes = get_max_size_per_gpu<value_t>( mpi.comm_world() );
    mpi.comm_world().barrier();
    t2.record();
    auto execution_time = t2.elapsed_time(t1); 
    log.info_f("init mpi and cuda: %lf ms.", execution_time);

    t1.record();
    auto mysize = sizes[myid];
    int nn = static_cast<int>(std::floor(std::pow(mem_factor*mysize, 1.0/3.0) ));
    vec_t vec3(nn,nn,nn);
    rect_t rect_l = rect_t(vec_t::make_zero(), vec3 );
    array_t vec_loc;
    vec_loc.init( vec3 );
    log.info_all_f("sizes are: %i %i %i, log_2(total size) = %.1f", nn,nn,nn, std::log( double(nn)*double(nn)*double(nn) )/std::log(2) );
    auto myidp1 = myid + 1;
    //functors::fill_values_ker<big_ordinal, value_t, dim, array_t>(rect_l, myidp1, vec_loc );
    functors::fill_values<for_each_t, big_ordinal, value_t, dim, array_t>(for_each, rect_l, myidp1, vec_loc );

    mpi.comm_world().barrier();
    t2.record();
    execution_time = t2.elapsed_time(t1); 
    log.info_f("init and fill arrays: %lf ms.", execution_time);
    //build reference reduce_sum
    value_t refernce_reduce_sum = 0;
    std::size_t total_sum = 0;
    {
        std::size_t mlt = 0;
        std::vector<std::size_t> volumes_ref;
        std::vector<std::size_t> volumes;
        std::transform(sizes.cbegin(), sizes.cend(), std::back_inserter(volumes_ref), [&mlt](std::size_t val)
                    {
                        auto nn = static_cast<std::size_t>(std::floor(std::pow(mem_factor*val, 1.0/3.0) ));
                        mlt++;
                        return nn*nn*nn*mlt; 
                    }
        );
        std::transform(sizes.cbegin(), sizes.cend(), std::back_inserter(volumes), [](std::size_t val)
                    {
                        auto nn = static_cast<std::size_t>(std::floor(std::pow(mem_factor*val, 1.0/3.0) ));
                        return nn*nn*nn; 
                    }
        );        
        refernce_reduce_sum = std::reduce(volumes_ref.begin(), volumes_ref.end());
        total_sum = std::reduce(volumes.begin(), volumes.end());


    }
    log.info_f("running on total_size = %le", static_cast<double>(total_sum) );

    timer_t t1_sum,t2_sum;
    timer_t t1_max,t2_max;
    timer_t t1_min,t2_min;
    value_t res_sum = 0;
    value_t res_max = 0;
    value_t res_min = 0;
    t1_sum.record();
    for(int j = 0; j<number_of_reps;j++)
    {

        res_sum = vec_ops.all_reduce_sum(vec_loc);
        mpi.comm_world().barrier();
    }
    t2_sum.record();
    execution_time = t2_sum.elapsed_time(t1_sum); 
    log.info_f("all_reduce_sum time: %lf ms.", execution_time/number_of_reps);
    auto check_sum = tests::check_test_to_eps(res_sum-refernce_reduce_sum);
    log.info_f("all_reduce_sum = %le, ref = %le,  %s", res_sum, refernce_reduce_sum, check_sum.first.c_str() );
    
    t1_max.record();    
    for(int j = 0; j<number_of_reps;j++)
    {
        res_max = vec_ops.all_reduce_max(vec_loc);
        mpi.comm_world().barrier();
    }
    t2_max.record();
    execution_time = t2_max.elapsed_time(t1_max);  
    log.info_f("all_reduce_max time: %lf ms.", execution_time/number_of_reps);   
    auto check_max = tests::check_test_to_zero(res_max-nproc);
    log.info_f("all_reduce_max = %.0lf, %s", res_max, check_max.first.c_str() );

    t1_min.record();
    for(int j = 0; j<number_of_reps;j++)    
    {
        res_min = vec_ops.all_reduce_min(vec_loc);
        mpi.comm_world().barrier();
    }
    t2_min.record();
    execution_time = t2_min.elapsed_time(t1_min);  
    log.info_f("all_reduce_min time: %lf ms.", execution_time/number_of_reps);   
    auto check_min = tests::check_test_to_zero(res_min-1);
    log.info_f("all_reduce_min = %.0lf, %s", res_min, check_min.first.c_str() );

    log.info("DONE");
    return !(check_sum.second&check_max.second&check_min.second);
}