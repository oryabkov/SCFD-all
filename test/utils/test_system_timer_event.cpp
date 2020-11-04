
#include <chrono>
#include <thread>
#include <iostream>
#include <scfd/utils/system_timer_event.h>

using namespace scfd::utils;

int main(int argc, char const *args[])
{
    system_timer_event  e1, e2;

    /// These are not working becase timer is Noncopyable and Nonmoveable
    //system_timer_event  e3(e1);
    //system_timer_event  e3(std::move(e1));
    //e1 = e2;
    //e1 = std::move(e2);
    
    e1.record();

    std::this_thread::sleep_for(std::chrono::seconds(2));

    e2.record();

    std::cout << "elapsed_time = " << e2.elapsed_time(e1) << " ms" << std::endl;

    return 0;
}