// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.


#include <iostream>
#include <scfd/arrays/last_index_fast_arranger.h>

using namespace scfd::arrays;

int main(int argc, char const *argv[])
{
    last_index_fast_arranger<10>               test1;
    last_index_fast_arranger<10,10>            test2;
    last_index_fast_arranger<10,10,5>          test3;
    last_index_fast_arranger<10,10,5,6>        test4;
    last_index_fast_arranger<10,10,5,6,7>      test5;
    last_index_fast_arranger<10,10,5,6,7,8>    test6;
    last_index_fast_arranger<10,10,5,6,7,8,9>  test7;

    std::cout << test2.calc_lin_index(0,0) << " "
              << test2.calc_lin_index(1,0) << " "
              << test2.calc_lin_index(2,0) << " "
              << test2.calc_lin_index(3,0) << " "
              << test2.calc_lin_index(4,0) << " " 
              << test2.calc_lin_index(0,1) << " "
              << test2.calc_lin_index(0,2) << " "
              << test2.calc_lin_index(0,3) << " "
              << test2.calc_lin_index(0,4) << " "
              << test2.calc_lin_index(1,2) << " "
              << test2.calc_lin_index(2,2) << " "
              << test2.calc_lin_index(3,2) << " " << std::endl;
    
    std::cout << test3.calc_lin_index(0,0,0) << " "
              << test4.calc_lin_index(0,0,0,0) << " "
              << test5.calc_lin_index(0,0,0,0,0) << " "
              << test6.calc_lin_index(0,0,0,0,0,0) << " "
              << test7.calc_lin_index(0,0,0,0,0,0,0) << " " << std::endl;

    return 0;
}