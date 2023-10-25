
#include <scfd/utils/log_mpi.h>

using namespace scfd::utils;

int main(int argc, char const *args[])
{
    log_mpi   log;

    log.info_f("test info format rank*2 = %d", log.comm_rank()*2);
    log.info_all_f("test info_all format rank*3 = %d", log.comm_rank()*3);
    
    return 0;
}