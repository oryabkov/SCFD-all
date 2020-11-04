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


#include <array>
#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_base.h>
#include <scfd/arrays/last_index_fast_arranger.h>

using namespace scfd::arrays;

typedef scfd::memory::host                            memory_t;

struct Vec
{

};

int main(int argc, char const *argv[])
{
    tensor_base<float,memory_t,last_index_fast_arranger,2,2,0,2>    array;
    Vec     v;
    //array.get_(v, 0,placeholder{},2,0);
    
    return 0;
}