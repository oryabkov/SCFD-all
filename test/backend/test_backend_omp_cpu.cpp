#include <complex>
#include <iostream>
#include <type_traits>
#include <scfd/utils/log_std.h>
#include <scfd/static_vec/vec.h>
#include <scfd/backend/omp.h>
#include <scfd/arrays/tensorN_array_nd.h>

#define PLATFORM_OMP
#include <scfd/backend/backend.h>

template<class Idx, class Vec>
struct func_nd3
{
    func_nd3(Vec& _f) : f(_f) {}
    Vec f;
    void operator()(const Idx& idx) const
    {
        f(idx) = {0, static_cast<double>(idx[1])};
    }
};

template<class Vec>
struct func
{
    func(Vec& _f) : f(_f) {}
    Vec f;
    void operator()(const std::size_t& idx) const
    {
        f(idx) = idx;
    }
};


int main(int argc, char const *argv[])
{
    using T = std::complex<double>;
    using log_t = scfd::utils::log_std;
    using backend_t = scfd::backend::omp;
    using memory_t = backend_t::memory_type;
    using for_each_t = backend_t::for_each_type<int>;
    using for_each_nd_t = backend_t::for_each_nd_type<3>;
    using reduce_t = backend_t::reduce_type;
    
    using backend_def_t = scfd::backend::current;

    if(!std::is_same<backend_t, backend_def_t>::value)
    {
        std::cout << "FAILED BACKEND TYPE CHECK" << std::endl;
        return 10;
    }

    using array_t = scfd::arrays::tensor0_array_nd<T, 1, memory_t>;
    using array3_t = scfd::arrays::tensor0_array_nd<T, 3, memory_t>;
    using idx3_t = scfd::static_vec::vec<int,3>;
    using rect_t = scfd::static_vec::rect<int, 3>;

    log_t log;

    for_each_t for_each;
    for_each_nd_t for_each_nd;
    reduce_t reduce;

    int SZ_X = 10, SZ_Y = 10, SZ_Z = 10;
    int SZ = SZ_X*SZ_Y*SZ_Z;
    array3_t f3;
    f3.init(idx3_t(SZ_X,SZ_Y,SZ_Z));
    rect_t range(idx3_t(0,0,0), idx3_t(SZ_X,SZ_Y,SZ_Z));
    for_each_nd(func_nd3<idx3_t, array3_t>(f3), range);
    //TODO! All reduces should accept scfd::arrays!
    //TODO! add tests for reduces!
    auto res = reduce(SZ, f3.raw_ptr(), {0,0});
    T res_ref = {0, 10*10*(9*(9+1))/2 };
    reduce.wait();
    auto diff = std::norm(res - res_ref);
    std::cout << "res = " << res << ", res_ref = " << res_ref << ", ||diff|| = " << diff <<  std::endl;
    
    array_t f1;
    f1.init(SZ);
    for_each(func<array_t>(f1), SZ);
    res = reduce(SZ, f1.raw_ptr(), {0,0});
    res_ref = {(SZ-1)*SZ*0.5, 0};
    reduce.wait();
    diff += std::norm(res - res_ref);
    std::cout << "res = " << res << ", res_ref = " << res_ref << ", ||diff|| = " << diff <<  std::endl;

    if(diff > 1.0e-12)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }

}

