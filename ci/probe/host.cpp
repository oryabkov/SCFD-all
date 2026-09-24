#include <omp.h>

#include <iostream>

int main()
{
    int sum = 0;
#pragma omp parallel for reduction( + : sum )
    for ( int i = 0; i < 8; ++i )
        sum += i;
    if ( sum != 28 )
        return 1;
    std::cout << "CPU/OpenMP is usable (maximum threads: " << omp_get_max_threads() << ")\n";
    return 0;
}
