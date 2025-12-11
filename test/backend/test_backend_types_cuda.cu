#include <type_traits>

#include <scfd/backend/serial_cpu.h>
#include <scfd/backend/cuda.h>
#define PLATFORM_CUDA
#include <scfd/backend/backend.h>

int main(int argc, char const *argv[])
{
    using backend_t = scfd::backend::cuda;
    using backend_def_t = scfd::backend::current;

    using memory_t = scfd::backend::memory;
    using for_each_t = scfd::backend::for_each<int>;
    using for_each_nd_t = scfd::backend::for_each_nd<3>;
    using reduce_t = scfd::backend::reduce;


    if(!std::is_same<backend_t, backend_def_t>::value)
    {
        std::cout << "FAILED 10" << std::endl;
        return 10;
    }
    else if(!std::is_same<backend_t::memory_type, memory_t>::value)
    {
        std::cout << "FAILED 11" << std::endl;
        return 11;
    }
    else if(!std::is_same<backend_t::for_each_type<int>, for_each_t>::value)
    {
        std::cout << "FAILED 12" << std::endl;
        return 12;
    } 
    else if(!std::is_same<backend_t::for_each_nd_type<3>, for_each_nd_t>::value)
    {
        std::cout << "FAILED 13" << std::endl;
        return 13;
    } 
    else if(!std::is_same<backend_t::reduce_type, reduce_t>::value)
    {
        std::cout << "FAILED 14" << std::endl;
        return 14;
    }            
    else
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }

    return 0;
}
