
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <scfd/memory/cuda.h>

int main(int argc, char const *args[])
{
    thrust::device_vector<int> d_v(10,2);

    auto thrust_dev_ptr_1 = scfd::memory::thrust_ptr_cast<scfd::memory::cuda_device,int>(thrust::raw_pointer_cast(&d_v[0]));
    auto thrust_dev_ptr_2 = scfd::memory::thrust_ptr_cast<scfd::memory::cuda_device>(thrust::raw_pointer_cast(&d_v[0]));
    std::cout << "reduce1 = " << thrust::reduce(thrust_dev_ptr_1,thrust_dev_ptr_1+10) << std::endl;
    std::cout << "reduce2 = " << thrust::reduce(thrust_dev_ptr_2,thrust_dev_ptr_2+10) << std::endl;
    
    return 0;
}