#include <type_traits>

#include <scfd/backend/serial_cpu.h>
#include <scfd/backend/cuda.h>
#define PLATFORM_CUDA
#include <scfd/backend/backend.h>

int main(int argc, char const *argv[])
{
    using backend_t = scfd::backend::cuda;
    using backend_def_t = scfd::backend::selection;

    if(!std::is_same<backend_t, backend_def_t>::value)
    {
        std::cout << "FAILED" << std::endl;
        return 10;
    }
    else
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }

    return 0;
}
