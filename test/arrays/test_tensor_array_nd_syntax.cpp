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


#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT

#include <iostream>
#include <array>
#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/arrays/tensor_array_nd_view.h>
#include <scfd/arrays/last_index_fast_arranger.h>

using namespace scfd::arrays;

typedef scfd::memory::host                                  memory_t;

struct Vec
{

};

int main(int argc, char const *argv[])
{
    float               x,y;
    vec<int,3>          sz3;

    tensor_array_nd<float,3,memory_t,last_index_fast_arranger,2,2,1,2>  array_3d;
    array_3d.init(100,10,10, 3);
    //array_3d.init(vec<int,3>(100,10,10), 3);
    //array_3d.init(vec<int,3>(100,10,10), vec<int,3>(0,0,0), 3, 0);
    std::cout << "array_3d.size() = " << array_3d.size() << std::endl;
    std::cout << "array_3d.total_size() = " << array_3d.total_size() << std::endl;
    sz3 = array_3d.size_nd();
    std::cout << "array_3d.size_nd() = " << sz3[0] << "," << sz3[1] << "," << sz3[2] << std::endl;
    x = array_3d(3,0,0, 0,0,0,0);
    scfd::static_vec::vec<float,2>     v;
    array_3d.get_vec(v, 3,0,0, 0,0,0,placeholder{});
    //array_3d.get_<int,int,int,int,int,int,int>(0,1,2, 0,0,0,placeholer);
    
    tensor_array_nd<float,1,memory_t,last_index_fast_arranger,2,2,1,2>      array_1d;
    array_1d.init(100,3);
    std::cout << "array_1d.size() = " << array_1d.size() << std::endl;
    std::cout << "array_1d.total_size() = " << array_1d.total_size() << std::endl;
    char ind0 = 3;
    x = array_1d(3UL, 0,0,0,0);
    y = array_1d(ind0, 0,0,0,0);

    scfd::static_vec::vec<int,1>       idx(3);
    scfd::static_vec::vec<float,2>     v1;
    array_1d.get_vec(v1,idx,0,0,0,placeholder{});
    array_1d.get_vec(v1,3,0,0,0);

    v1 = array_1d.get_vec(idx,0,0,0,placeholder{});
    v1 = array_1d.get_vec(idx,0,0,0);
    v1 = array_1d.get_vec(3,0,0,0);

    auto    array_1d_view = array_1d.create_view();
    array_1d_view.free();

    tensor_array_nd<float,1,memory_t,last_index_fast_arranger,2,2,0,2>::view_type   view_zero_tensor;

    tensor_array_nd<float,3,memory_t,last_index_fast_arranger,2,2,0,2>  array_3d_zero_tensor;
    array_3d_zero_tensor.init(100,10,10, 3);
    std::cout << "array_3d_zero_tensor.size() = " << array_3d_zero_tensor.size() << std::endl;
    std::cout << "array_3d_zero_tensor.total_size() = " << array_3d_zero_tensor.total_size() << std::endl;
    sz3 = array_3d_zero_tensor.size_nd();
    std::cout << "array_3d_zero_tensor.size_nd() = " << sz3[0] << "," << sz3[1] << "," << sz3[2] << std::endl;
    tensor_array_nd<float,1,memory_t,last_index_fast_arranger,2,2,0,2>  array_1d_zero_tensor;
    array_1d_zero_tensor.init(100,3);
    std::cout << "array_1d_zero_tensor.size() = " << array_1d_zero_tensor.size() << std::endl;
    std::cout << "array_1d_zero_tensor.total_size() = " << array_1d_zero_tensor.total_size() << std::endl;

    return 0;
}