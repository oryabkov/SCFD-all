// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_HOST_H__
#define __SCFD_INVERT_UPPER_TRIANGULAR_DENSE_MATRIX_HOST_H__

namespace scfd
{

/// r_mat is upper triangular dense square szXsz matrix in col-major format 
/// which will be rewritten by temporal data.
/// r_mat lower part is irrelevant and also would be ruined
/// r_inv_mat is matrix of the same size that will contain inv(r_mat)
/// r_diag_inv, mat_tmp_col_i are temporal buffers for vectors of size sz
template<class T>
inline void invert_upper_triangular_dense_matrix_host(int sz, T* r_mat, T* r_diag_inv, T *mat_tmp_col_i, T* r_inv_mat)
{
    for (int i = 0;i < sz;++i)
    {
        for (int j = 0;j < sz;++j)
        {
            r_inv_mat[i+sz*j] = (i == j? T(1) : T(0));
        }
    }
    for (int i = 0;i < sz;++i)
    {
        r_diag_inv[i] = T(1)/r_mat[i+sz*i];
    }
    //std::cout << "here2" << std::endl;
    for (int j = 0;j < sz;++j)
    {
        for (int i = 0;i <= j;++i)
        {
            r_mat[i+sz*j] *= r_diag_inv[i];
        }
        r_inv_mat[j+sz*j] *= r_diag_inv[j];
    }
    //std::cout << "here3" << std::endl;
    for (int i = sz-1;i >= 0;--i)
    {            
        for (int i1 = 0;i1 < i;++i1)
        {
            mat_tmp_col_i[i1] = r_mat[i1+sz*i];
        }
        //std::cout << "here4: i = " << i << std::endl;
        #pragma omp parallel for
        for (int j = i;j < sz;++j)
        {
            for (int i1 = 0;i1 < i;++i1)
            {
                //make r_mat(i1,i) to be 0
                T mul = mat_tmp_col_i[i1];
                r_inv_mat[i1+sz*j] -= r_inv_mat[i+sz*j]*mul;
            }
        }
        /*std::cout << "i = " << i << std::endl;
        for (int ii = 0;ii < i;++ii)
        {
            std::cout << mat_tmp_col_i[ii] << std::endl;
        }*/
        /*for (int ii = 0;ii < sz;++ii)
        {
            for (int jj = 0;jj < sz;++jj)
            {
                std::cout << r_inv_mat(ii,jj) << " ";
            }
            std::cout << std::endl;
        }*/
    }
}

} // namespace scfd

#endif
