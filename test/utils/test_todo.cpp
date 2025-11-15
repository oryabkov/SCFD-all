#include <scfd/utils/todo.h>
// #include <scfd/utils/todo_kernel.h>
#include <iostream>
#include <cassert>




int main(int argc, char const *argv[])
{
    std::cout << "this file deomstrates the applicaiton of TODO in a CPP code." << std::endl;
    
    try
    {
        SCFD_TODO("this is a TODO text cought by try-catch block with runtime_error. This TODO cannot be used in device kernels!");
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
    }
    SCFD_ATODO("this is a global TODO assertion that can't be cought. This TODO can be used in device kernels! ");



    return 0;
}