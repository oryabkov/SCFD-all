#include <iostream>
#include <omp.h>

int main()
{
    int iterations = 0;
    int sum        = 0;
#pragma omp parallel for reduction( + : iterations, sum )
    for ( int i = 0; i < 16; i++ )
    {
        const int threadID = omp_get_thread_num();
        ++iterations;
        sum += i;
#pragma omp critical
        {
            std::cout << "Thread " << threadID << " reporting" << std::endl;
        }
    }
    const bool passed = iterations == 16 && sum == 120;
    std::cout << "OpenMP worksharing test: " << ( passed ? "PASS" : "FAIL" ) << std::endl;
    return passed ? 0 : 1;
}
