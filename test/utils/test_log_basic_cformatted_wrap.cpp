
#include <scfd/utils/log_std_basic.h>
#include <scfd/utils/log_basic_cformatted_wrap.h>

using namespace scfd::utils;

using log_format_wrap_t = log_basic_cformatted_wrap<log_std_basic>;

int main(int argc, char const *args[])
{
    log_std_basic       log_basic;
    log_format_wrap_t   log_format_wrap(&log_basic);

    log_format_wrap.info_f("test %d", 100);
    
    return 0;
}