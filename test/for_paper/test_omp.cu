#include <iostream>
#include <omp.h>

int main()
{
    int i;
    int threadID = 0;
    #pragma omp parallel for private(i, threadID)
    for(i = 0; i < 16; i++ )
    {
        threadID = omp_get_thread_num();
        #pragma omp critical
        {
            std::cout << "Thread " << threadID << " reporting" << std::endl;
        }
    }
    return 0;
}