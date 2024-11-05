
#include <iostream>
#include <string>
#include <scfd/utils/format_json.h>

int main(int argc, char **args)
{
    nlohmann::json 
        j = 
        {
            {"one", 1}, {"two", 2},
            {
                "three", 
                {
                    {"sub_one","one_str"}, {"sub_two",2.2}, 
                    {"sub_three",{{"subsub_one",true}}},
                    {"sub_four",{1,2,3}}
                }
            }
        };
    scfd::utils::format_json(j, std::cout, 0);

    return 0;
}