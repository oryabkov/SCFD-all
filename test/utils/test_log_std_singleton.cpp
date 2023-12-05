
#include <scfd/utils/log_std.h>
#include <scfd/utils/log_std_singleton_impl.h>

using namespace scfd::utils;

int main(int argc, char const *args[])
{
    log_std  log;
    log_std::set_inst(&log);

    log_std::inst().info_f("test %d", 100);
    
    return 0;
}