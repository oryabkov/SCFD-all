#include <scfd/utils/log_std.h>
#include <scfd/static_vec/vec.h>
#include <scfd/backend/cuda.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/utils/device_tag.h>

#include <scfd/utils/todo.h>


namespace detail
{

template<class Vec, bool UseTodo>
struct func
{
    func(Vec& _f) : f(_f) {}
    Vec f;
    __DEVICE_TAG__ void operator()(const std::size_t& idx) const
    {
        if constexpr (UseTodo)
        {
            SCFD_ATODO("do some work in kernel.");
            f(idx) = idx;
        }
        else
        {
            f(idx) = idx;
        }
    }
};

}



int main(int argc, char const *argv[])
{
    using log_t = scfd::utils::log_std;
    using backend_t = scfd::backend::cuda;
    using memory_t = backend_t::memory_type;
    using for_each_t = backend_t::for_each_type<int>;
    using array_t = scfd::arrays::tensor0_array_nd<int, 1, memory_t>;

    log_t log;
    for_each_t for_each;
    
    int SZ = 1000;

    array_t f1;
    f1.init(SZ);
    for_each(detail::func<array_t, false>(f1), SZ);
    for_each.wait();
    log.info("executed kernel without TODO");
    log.info("executing kernel with TODO...");
    for_each(detail::func<array_t, true>(f1), SZ);
    for_each.wait();

    return 0;
}