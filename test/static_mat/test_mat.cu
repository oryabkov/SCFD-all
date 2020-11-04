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

#include <cuda.h>
#include <iostream>
#include <scfd/static_mat/mat.h>

using namespace scfd::static_mat;

template<typename T, class T_mat>
__host__ __device__ T some_func(T_mat& m2)
{
    return (m2(0,0)+m2(1,1));


}


template<typename T, class Mat1,class Mat2>
__global__ void ket_test(Mat1 m1, Mat2 m2)
{
    auto m3 = m1*m2;

    for (int i = 0;i < 3;++i) {
        for (int j = 0;j < 4;++j)
            printf("%f ", m3(i,j));
        printf("\n");
    }
    printf("ker = %le \n",some_func<T, Mat2>(m2));

   // printf("test\n");   
}

/*int main(int argc, char const *argv[])
{
    cudaSetDevice(1);

    mat<float,3,3>    m1;
    //mat<int,3,4>      m_int;
    mat<float,3,4>    m2,m3;

    std::cout << "sizeof(vec<float,3>) = " << sizeof(mat<float,3,4>) << std::endl;
    std::cout << "sizeof(vec<int,3>) = " << sizeof(mat<int,3,3>) << std::endl;

    m1(0,0) = 1.; m1(0,1) = 3.; m1(0,2) = 7.;
    m1(1,0) = 0.; m1(1,1) = 1.; m1(1,2) = 0.;
    m1(2,0) = 8.; m1(2,1) = 5.; m1(2,2) = 1.;

    m2(0,0) = 1.; m2(0,1) = 2. ; m2(0,2) = 3. ; m2(0,3) = 4.;
    m2(1,0) = 5.; m2(1,1) = 6. ; m2(1,2) = 7. ; m2(1,3) = 8.;
    m2(2,0) = 9.; m2(2,1) = 10.; m2(2,2) = 11.; m2(2,3) = 12.;

    m3 = m1*m2;
    for (int i = 0;i < 3;++i) {
        for (int j = 0;j < 4;++j)
            std::cout << m3(i,j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    m3 += m2;
    for (int i = 0;i < 3;++i) {
        for (int j = 0;j < 4;++j)
            std::cout << m3(i,j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    m3 = m2+m3;
    for (int i = 0;i < 3;++i) {
        for (int j = 0;j < 4;++j)
            std::cout << m3(i,j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    m3 = m2-m3;
    for (int i = 0;i < 3;++i) {
        for (int j = 0;j < 4;++j)
            std::cout << m3(i,j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    auto m4 = m3.transposed();
    for (int i = 0;i < 4;++i) {
        for (int j = 0;j < 3;++j)
            std::cout << m4(i,j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    ket_test<float, mat<float,3,3> , mat<float,3,4> ><<<1,1>>>(m1,m2);

    cudaDeviceSynchronize();

    return 0;
}*/
