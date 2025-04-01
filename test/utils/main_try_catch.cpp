
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <scfd/utils/log.h>
#include <scfd/utils/main_try_catch_macro.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/safe_call.h>

void nested_error1(const std::string &msg)
{
    throw std::runtime_error("nested_error1 : " + msg);
}

void nested_error2(const std::string &msg)
{
    SCFD_SAFE_CALL( nested_error1(msg) );
}

int main(int argc, char **args)
{
    if (argc < 2) {
        std::cout << "USAGE: " << std::string(args[0]) << " <block_number>" << std::endl;
        return 0;
    }

    scfd::utils::log_std  log;
    SCFD_USE_MAIN_TRY_CATCH(log)  

    int block_number = atoi(args[1]);

    SCFD_MAIN_TRY("test block 1")
    if (block_number == 1) 
    {
        SCFD_SAFE_CALL( nested_error2("error block 1") );
    }
    SCFD_MAIN_CATCH(1)

    SCFD_MAIN_TRY("test block 2")
    if (block_number == 2) throw std::runtime_error("error block 2");
    SCFD_MAIN_CATCH(2)

    SCFD_MAIN_TRY("test block 3")
    if (block_number == 3) throw std::runtime_error("error block 3");
    SCFD_MAIN_CATCH(3)

    return 0;
}