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
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>

#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_base.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/arrays/arrays_config.h>
#include <scfd/static_vec/vec.h>
#include <scfd/arrays/arranger_base.h>

namespace scfd
{
namespace arrays
{
using static_vec::vec;


template<size_t I>
struct axis_size {
    template<typename... Ts>
    static std::size_t get(const Ts&... sizes) { return std::get<I>(std::forward_as_tuple(sizes...)); }
};


template<ordinal_type... Dims>
struct custom_index_arranger : public arranger_base<ordinal_type,Dims...>
{
};


template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type First, ordinal_type Second>
struct custom_index_arranger<Dim0, Dim1, First, Second> : public arranger_base<ordinal_type,Dim0,Dim1>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1)const
    {

        
        #ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
                
                const auto first_axis = axis_size<First>::get( this->template get_dim<0>(), this->template get_dim<1>() );
                return first_axis*axis_size<First>::get(i0 - this->template get_index0<0>(), i1 - this->template get_index0<1>()) + axis_size<Second>::get(i0, i1); //(i0 - this->template get_index0<0>())*this->template get_dim<1>() + i1 - this->template get_index0<1>();


        #else
                const auto first_axis = axis_size<Second>::get( this->template get_dim<0>(), this->template get_dim<1>() );
                return first_axis*axis_size<First>::get(i0, i1) + axis_size<Second>::get(i0, i1); //(i0)*this->template get_dim<1>() + i1;
        #endif
        
    }
};

template<ordinal_type Dim0, ordinal_type Dim1, ordinal_type Dim2, ordinal_type First, ordinal_type Second, ordinal_type Third>
struct custom_index_arranger<Dim0,Dim1,Dim2,First, Second, Third> : public arranger_base<ordinal_type,Dim0,Dim1,Dim2>
{
    __DEVICE_TAG__ ordinal_type calc_lin_index(ordinal_type i0, 
                                               ordinal_type i1, 
                                               ordinal_type i2)const
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        printf("TODO! Implement on demand!");
        exit(0);
        return                              ((i2 - this->template get_index0<2>())*
                this->template get_dim<1>() + i1 - this->template get_index0<1>())*
                this->template get_dim<0>() + i0 - this->template get_index0<0>();
#else
 
        return axis_size<First>::get(i0, i1, i2) + axis_size<First>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>() )*( axis_size<Second>::get(i0, i1, i2) +  axis_size<Second>::get( this->template get_dim<0>(), this->template get_dim<1>(), this->template get_dim<2>() )* axis_size<Third>::get(i0, i1, i2) );
        // return  ((i2)*this->template get_dim<1>() + i1)*this->template get_dim<0>() + i0;
#endif
    }
};

}
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


int main(int argc, char const *argv[])
{
    using memory_t = scfd::memory::host;
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
    std::vector<T> vec_check(Nx*Ny, 0);

    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    {
        // Ny*x+y
        auto val =  Ny*j+k;
        // std::cout << val << " ";
        vec_check.at(Ny*j+k) = val;
        array01(j,k) = val;
        array10(j,k) = val;
    }

    bool ok_flag2 = true;
    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    {
        auto diff01 = std::abs(vec_check.at(Ny*j+k) - array01(j,k) );
        auto diff10 = std::abs(vec_check.at(Ny*j+k) - array10(j,k) );

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

    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    for(int l=0;l<Nz;l++)
    {
        // Ny*x+y
        auto val = (Ny*j+k)*Nz+l;
        // std::cout << val << " ";
        vec_check3.at( val ) = val;
        array012(j,k,l) = val;
        array210(j,k,l) = val;
        array120(j,k,l) = val;
        array102(j,k,l) = val;
        array201(j,k,l) = val;
        array021(j,k,l) = val;                
    }


    auto check_diff = [](const std::vector<int>& idxs, auto& v1, auto& v2, const std::string& name, bool& ok_flag)
    {
        auto diff = std::abs(v1 - v2);
        if(diff > std::numeric_limits<T>::epsilon() )
        {
            std::cerr << "error for array01 at: ( ";
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
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array012(j,k,l), "012", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array210(j,k,l), "210", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array120(j,k,l), "120", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array102(j,k,l), "102", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array201(j,k,l), "201", ok_flag3 );
        check_diff({j,k,l}, vec_check3.at((Ny*j+k)*Nz+l), array021(j,k,l), "021", ok_flag3 );


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