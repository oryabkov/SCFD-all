#ifdef PLATFORM_OMP
#    include <omp.h>
#endif

#include <iostream>

int main()
{
    int sum = 0;
#ifdef PLATFORM_OMP
#    pragma omp parallel for reduction( + : sum )
#endif
    for ( int i = 0; i < 8; ++i )
        sum += i;
    if ( sum != 28 )
        return 1;
#ifdef PLATFORM_OMP
    std::cout << "OpenMP is usable (maximum threads: " << omp_get_max_threads() << ")\n";
#else
    std::cout << "SERIAL execution is usable\n";
#endif
    return 0;
}
