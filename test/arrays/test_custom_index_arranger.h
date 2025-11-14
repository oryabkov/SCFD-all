// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_TEST_CUSTOM_INDEX_ARRANGER_H__
#define __SCFD_TEST_CUSTOM_INDEX_ARRANGER_H__


#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>

#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_base.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/arrays/arrays_config.h>

#include "custom_index_arranger.h"






namespace scfd
{

namespace tests
{


namespace detail
{

template<class Idx, class Array1, class Array2>
struct copy_work
{
    copy_work(const Array1& _f1, Array2& _f2) : f1(_f1), f2(_f2) {}
    Array1 f1;
    Array2 f2;

    __DEVICE_TAG__ void operator()(const Idx& idx)
    {
        f2(idx) =  f1(idx);
    }

};



}



template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1>
using custom_arranger_01_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, 0, 1 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1>
using custom_arranger_10_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, 1, 0 >;


template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_012_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 0, 1, 2 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_120_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 1, 2, 0 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_210_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 2, 1, 0 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_102_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 1, 0, 2 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_201_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 2, 0, 1 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_021_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 0, 2, 1 >;


// template<int I>
// std::size_t ss(int Nx, int Ny, int Nz)
// {
//     return scfd::arrays::axis_size<I>::get(Ny*Nz, Nx*Nz, Nx*Ny);
// };
template<int I>
std::size_t ff(int Nx, int Ny, int Nz)
{
     return scfd::arrays::axis_size<I>::get(Nx,Ny,Nz);
}


template<class Memory, class ForEach3>
int check()
{
    using memory_t = Memory;
    using for_each_t = ForEach3;
    using T = float;
    int Nx, Ny, Nz;

    //test tuple extraction (https://en.cppreference.com/w/cpp/utility/tuple/forward_as_tuple Constructs a tuple of references to the arguments in args suitable for forwarding as an argument to a function. The tuple has rvalue reference data members when rvalues are used as arguments, and otherwise has lvalue reference data members):
    std::cout << "tuple extraction 0: " << std::get<0>( std::forward_as_tuple(9, 10, 'A') ) << std::endl;
    std::cout << "tuple extraction 1: " << std::get<1>( std::forward_as_tuple(9, 10, 'A') ) << std::endl;
    std::cout << "tuple extraction 2: " << std::get<2>( std::forward_as_tuple(9, 10, 'A') ) << std::endl;

    std::cout << "checking for custom_index_arranger by hand: " << std::endl;

    Nx = 3; Ny = 5; Nz = 2;
    std::cout << "Nx: " << Nx << ", Ny: " << Ny << ", Nz: " << Nz << std::endl;
    std::cout << "f0: " << ff<0>(Nx,Ny,Nz) << std::endl;
    std::cout << "f1: " << ff<1>(Nx,Ny,Nz) << std::endl;
    std::cout << "f2: " << ff<2>(Nx,Ny,Nz) << std::endl;
    // index 201: z+Nz*(x+Nx*y) = z+Nz*x+Nz*Nx*y
    auto idx012 = ff<0>(2,3,1) + ff<0>(Nx,Ny,Nz)*(ff<1>(2,3,1) + ff<1>(Nx,Ny,Nz)*ff<2>(2,3,1));
    std::cout << "012 (2,3,1): " << idx012 << ", correct:" << 2+3*Nx+1*Nx*Ny << std::endl;

    auto idx201 = ff<2>(2,3,1) + ff<2>(Nx,Ny,Nz)*(ff<0>(2,3,1) + ff<0>(Nx,Ny,Nz)*ff<1>(2,3,1));
    std::cout << "201 (2,3,1): " << idx201 << ", correct:" << 1+2*Nz+3*Nz*Nx << std::endl;

    const int idx0 = 1;
    const int idx1 = 2;
    const int idx2 = 0;
    auto idx_req = ff<idx0>(2,3,1) + ff<idx0>(Nx,Ny,Nz)*(ff<idx1>(2,3,1) + ff<idx1>(Nx,Ny,Nz)*ff<idx2>(2,3,1));
    std::cout << "req (2,3,1): " << idx_req << ", correct:" << 3+Ny*(1 + Nz*2) << std::endl;

    std::cout << "==== 2D custom_index_arranger by hand ====" << std::endl;
    //test calling of custom arangers
    scfd::arrays::custom_index_arranger<10,10, 0, 1> test2_01;
    scfd::arrays::custom_index_arranger<10,10, 1, 0> test2_10;

    scfd::arrays::custom_index_arranger<10,9,8, 0, 1, 2> test3_012;
    scfd::arrays::custom_index_arranger<10,9,8, 1, 2, 0> test3_120;
    scfd::arrays::custom_index_arranger<10,9,8, 2, 0, 1> test3_201;    


    // scfd::arrays::custom_index_arranger<10,10,5> test3;
    // scfd::arrays::custom_index_arranger<10,10,5,6> test4;

    std::cout << test2_01.calc_lin_index(0,0) << " " << test2_01.calc_lin_index(10,0) << " " << test2_01.calc_lin_index(5,5) << " " << test2_01.calc_lin_index(0,10) << std::endl;
    std::cout << test2_10.calc_lin_index(0,0) << " " << test2_10.calc_lin_index(10,0) << " " << test2_10.calc_lin_index(5,5) << " " << test2_10.calc_lin_index(0,10) << std::endl;

    std::cout << test3_012.calc_lin_index(0,0,0) << " " << test3_012.calc_lin_index(10,0,0) << " " << test3_012.calc_lin_index(0,9,0) << " " << test3_012.calc_lin_index(0,0,8) << " " << test3_012.calc_lin_index(5,5,0) << " " << test3_012.calc_lin_index(0,5,5) << " " << test3_012.calc_lin_index(5,0,5) << " " << test3_012.calc_lin_index(5,5,5) << std::endl;
    std::cout << test3_120.calc_lin_index(0,0,0) << " " << test3_120.calc_lin_index(10,0,0) << " " << test3_120.calc_lin_index(0,9,0) << " " << test3_120.calc_lin_index(0,0,8) << " " << test3_120.calc_lin_index(5,5,0) << " " << test3_120.calc_lin_index(0,5,5) << " " << test3_120.calc_lin_index(5,0,5) << " " << test3_120.calc_lin_index(5,5,5) << std::endl;
    std::cout << test3_201.calc_lin_index(0,0,0) << " " << test3_201.calc_lin_index(10,0,0) << " " << test3_201.calc_lin_index(0,9,0) << " " << test3_201.calc_lin_index(0,0,8) << " " << test3_201.calc_lin_index(5,5,0) << " " << test3_201.calc_lin_index(0,5,5) << " " << test3_201.calc_lin_index(5,0,5) << " " << test3_201.calc_lin_index(5,5,5) << std::endl;

    std::cout << "==== 2D arrays ====" << std::endl;


    scfd::arrays::tensor_array_nd<T, 2, memory_t, custom_arranger_01_t> array01;
    scfd::arrays::tensor_array_nd<T, 2, memory_t, custom_arranger_01_t> array10;

    Nx = 10; Ny = 20;
    array01.init(Nx, Ny);
    array10.init(Nx, Ny);
    typename decltype(array01)::view_type array01_view(array01);
    typename decltype(array01)::view_type array10_view(array10);
    std::vector<T> vec_check(Nx*Ny, 0);

    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    {
        // Ny*x+y
        auto val =  Ny*j+k;
        // std::cout << val << " ";
        vec_check.at(Ny*j+k) = val;
        array01_view(j,k) = val;
        array10_view(j,k) = val;
    }
    array01_view.release(true);
    array10_view.release(true);

    array01_view.init(array01);
    array10_view.init(array10);

    bool ok_flag2 = true;
    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    {
        auto diff01 = std::abs(vec_check.at(Ny*j+k) - array01_view(j,k) );
        auto diff10 = std::abs(vec_check.at(Ny*j+k) - array10_view(j,k) );

        if(diff01 > std::numeric_limits<T>::epsilon() )
        {
            std::cerr << "error for array01 at:  (" << j << "," << k << ") with diff: " << diff01 << std::endl;
            ok_flag2 = false;
        }
        if(diff10 > std::numeric_limits<T>::epsilon() )
        {
            std::cerr << "error for array10 at: (" << j << "," << k << ") with diff: " << diff10 << std::endl;
            ok_flag2 = false;
        }        
    }
    if(ok_flag2)
    {
        std::cout << "ok" << std::endl;
    }

    std::cout << "==== 3D arrays ====" << std::endl;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_012_t> array012;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_120_t> array120;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_210_t> array210;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_102_t> array102;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_201_t> array201;
    scfd::arrays::tensor_array_nd<T, 3, memory_t, custom_arranger_021_t> array021;

    Nx = 3; Ny = 2; Nz = 4;
    array012.init(Nx, Ny, Nz);
    array120.init(Nx, Ny, Nz);
    array210.init(Nx, Ny, Nz);
    array102.init(Nx, Ny, Nz);
    array201.init(Nx, Ny, Nz);
    array021.init(Nx, Ny, Nz);                
    std::vector<T> vec_check3(Nx*Ny*Nz, 0);
    typename decltype(array012)::view_type array012_view(array012);
    typename decltype(array120)::view_type array120_view;
    typename decltype(array210)::view_type array210_view;
    typename decltype(array102)::view_type array102_view;
    typename decltype(array201)::view_type array201_view;
    typename decltype(array021)::view_type array021_view;



    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    for(int l=0;l<Nz;l++)
    {
        // Ny*x+y
        auto val = (Ny*j+k)*Nz+l;
        // std::cout << val << " ";
        vec_check3.at( val ) = val;
        array012_view(j,k,l) = val;


    }
    array012_view.release(true);

    using idx_t = scfd::static_vec::vec<int, 3>;
    for_each_t for_each;
    scfd::static_vec::rect<int, 3> range_T(idx_t(0,0,0), idx_t(Nx, Ny, Nz));
    
    for_each( detail::copy_work<idx_t, decltype(array012), decltype(array120)>(array012, array120), range_T);
    for_each.wait();
    for_each( detail::copy_work<idx_t, decltype(array012), decltype(array210)>(array012, array210), range_T);
    for_each.wait();    
    for_each( detail::copy_work<idx_t, decltype(array012), decltype(array102)>(array012, array102), range_T);
    for_each.wait();
    for_each( detail::copy_work<idx_t, decltype(array012), decltype(array201)>(array012, array201), range_T);
    for_each.wait();
    for_each( detail::copy_work<idx_t, decltype(array012), decltype(array021)>(array012, array021), range_T);
    for_each.wait();    

    array012_view.init(array012);
    array210_view.init(array210);
    array120_view.init(array120);
    array102_view.init(array102);
    array201_view.init(array201);
    array021_view.init(array021);

    auto check_diff = [](const std::vector<int>& idxs, auto& v1, auto& v2, const std::string& name, bool& ok_flag)
    {
        auto diff = std::abs(v1 - v2);
        if(diff > std::numeric_limits<T>::epsilon() )
        {
            std::cerr << "error for array" << name << " at: ( ";
            for(auto& i: idxs)
            {
                std::cerr << i << " ";
            }
            std::cerr << ") with diff: " << diff << std::endl;
            ok_flag = false;
        }

    };

    bool ok_flag3 = true;
    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    for(int l=0;l<Nz;l++)
    {
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array012_view(j,k,l), "012", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array210_view(j,k,l), "210", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array120_view(j,k,l), "120", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array102_view(j,k,l), "102", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array201_view(j,k,l), "201", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array021_view(j,k,l), "021", ok_flag3 );
    }
    if(ok_flag3)
    {
        std::cout << "ok" << std::endl;
    }

    if(ok_flag3&&ok_flag2)
    {
        return 0;
    }
    else
    {
        return 1;
    }

}



}
}

#endif

