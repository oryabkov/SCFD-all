
#include <scfd/utils/init_cuda.h>
#include <scfd/external_libraries/cusolver_wrap.h>

scfd::cublas_wrap cublas_wrap_test;
scfd::cusolver_wrap cusolver_wrap_test(&cublas_wrap_test);

int main(int argc, char const *args[])
{
    scfd::utils::init_cuda_persistent();

    return 0;
}