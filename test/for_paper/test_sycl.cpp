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

#include <scfd/utils/system_timer_event.h>

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


using for_each_sycl_t = scfd::for_each::sycl_<>;
using mem_device_t = scfd::memory::sycl_device;
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


    __DEVICE_TAG__ void operator()(const int &idx) const
    {
        f_out_(idx,0) = f1_(idx,1)*f2_(idx,2)-f1_(idx,2)*f2_(idx,1);
        f_out_(idx,1) = -( f1_(idx,0)*f2_(idx,2)-f1_(idx,2)*f2_(idx,0) );
        f_out_(idx,2) = f1_(idx,0)*f2_(idx,1)-f1_(idx,1)*f2_(idx,0);

    }
};
template<> struct sycl::is_device_copyable<func_cross_prod<array_device_t>> : std::true_type {};

template<class ForEach, class Vec3>
void cross_prod_device(const std::size_t N, const Vec3& u, const Vec3& v, Vec3& w)
{

    ForEach for_each;
    for_each(func_cross_prod<Vec3>(u, v, w), 0, N);
    for_each.wait();

}

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " N iters" << std::endl;
        std::cout << "where: N is a size of R^{3XN}, iters is number of iterations for better measurements," << std::endl;
        return 1;
    }
    std::size_t N = std::atoi(argv[1]);
    std::size_t number_of_iters = std::atoi(argv[2]);

    std::size_t total_size = 3*N;

    std::random_device rd;
    std::mt19937 engine{ rd() };
    std::uniform_real_distribution<> dist(-100.0, 100.0);

    array_device_t u_dev, v_dev, cross_dev;
    array_host_t u_host, v_host, cross_host;

    u_dev.init(N); v_dev.init(N); cross_dev.init(N);
    u_host.init(N); v_host.init(N); cross_host.init(N);
    array_device_view_t u_dev_view(u_dev), v_dev_view(v_dev), cross_dev_view(cross_dev);
    array_host_t u_host_view(u_host), v_host_view(v_host), cross_host_view(cross_host);

    #pragma omp parallel for
    for(std::size_t j=0; j<N; j++ )
    {
        for(std::size_t k=0; k<3; k++)
        {
            T u_ = dist(engine);
            T v_ = dist(engine);
            u_dev_view(j,k) = u_;
            v_dev_view(j,k) = v_;
            cross_dev_view(j,k) = 0.0;
            u_host_view(j,k) = u_;
            v_host_view(j,k) = v_;
            cross_host_view(j,k) = 0.0;
        }
    }

	//WARM UP
	for(int it_ = 0; it_ < number_of_iters; it_++)
	{
	    cross_prod_device<for_each_sycl_t, array_device_t>(N, u_dev, v_dev, cross_dev);
	}
	std::vector<double> dev_tensor; dev_tensor.reserve(number_of_iters);
	for(int it_ = 0; it_ < number_of_iters; it_++)
	{
	    auto start = std::chrono::high_resolution_clock::now();
	    cross_prod_device<for_each_sycl_t, array_device_t>(N, u_dev, v_dev, cross_dev);
	    auto end = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
	    dev_tensor.push_back( elapsed_seconds.count() );
	}

	std::cout << "device tensor time = "
		  << std::accumulate(begin(dev_tensor), end(dev_tensor), 0.0)/number_of_iters
		  << "ms." << std::endl;

	std::string filename;
	filename = "execution_times_array_3d_cross_product_sycl.csv";
	std::fstream out_file{filename, out_file.out};

	if (!out_file.is_open())
	    std::cout << "failed to open " << filename << '\n';
	else
	{
	    out_file << "tensor" << std::endl;
	    for(int j = 0; j<number_of_iters; j++)
	    {
		out_file << dev_tensor.at(j) << std::endl;
	    }
	    out_file.close();
	}

    return 0;
}
