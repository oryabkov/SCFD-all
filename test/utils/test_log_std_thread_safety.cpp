
#include <vector>
#include <thread>
#include <scfd/utils/log_std.h>

using namespace scfd::utils;

int main(int argc, char const *args[])
{
    log_std       log;

    std::vector<std::thread> threads(5);

    for (std::size_t i = 0;i < threads.size();++i) 
    {
        threads[i] = std::thread
        (
            [&log,i]
            {
                log.info_f("test message from thread %d", i);
            }
        );
    }

    for (std::size_t i = 0;i < threads.size();++i) 
        threads[i].join();
    
    return 0;
}