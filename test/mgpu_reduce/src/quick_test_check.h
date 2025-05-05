// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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


#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__


#include <limits>
#include <utility>

namespace tests
{


template<class T>
std::pair<std::string, bool> check_test_to_eps(const T val)
{
    if( !std::isfinite(val) )
    {
        return {"\x1B[31mFAIL, NOT FINITE\033[0m", false};
    }
    else if(std::abs(val)>std::sqrt(std::numeric_limits<T>::epsilon()) )
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

template<class T>
std::pair<std::string, bool> check_test_to_zero(const T val)
{
    if( !std::isfinite(val) )
    {
        return {"\x1B[31mFAIL, NOT FINITE\033[0m", false};
    }
    else if(std::abs(val) != 0 )
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

std::pair<std::string, bool> check_to_bool(const bool val)
{
    if(!val)
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

}

#endif //__SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__