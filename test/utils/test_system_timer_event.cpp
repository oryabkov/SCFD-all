
#include <chrono>
#include <cmath>
#include <thread>
#include <iostream>
#include <string>
#include <scfd/utils/system_timer_event.h>

using namespace scfd::utils;

int main( int argc, char const *args[] )
{
    const bool         ci = argc == 2 && std::string( args[1] ) == "--ci";
    system_timer_event e1, e2;

    /// These are not working becase timer is Noncopyable and Nonmoveable
    //system_timer_event  e3(e1);
    //system_timer_event  e3(std::move(e1));
    //e1 = e2;
    //e1 = std::move(e2);

    e1.record();

    std::this_thread::sleep_for( std::chrono::milliseconds( ci ? 20 : 2000 ) );

    e2.record();

    const double elapsed = e2.elapsed_time( e1 );
    std::cout << "elapsed_time = " << elapsed << " ms" << std::endl;
    if ( ci )
    {
        if ( !std::isfinite( elapsed ) || elapsed < 5.0 )
            return 1;
        std::cout << "timer check passed" << std::endl;
    }

    return 0;
}
