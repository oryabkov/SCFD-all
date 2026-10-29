# Copyright (C) 2026 SCFD contributors

# This file is part of SCFD.

# SCFD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2 only of the License.

# SCFD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

add_executable(scalar_traits_test.bin test_scalar_traits.cpp)
target_compile_features(scalar_traits_test.bin PRIVATE cxx_std_17)
scfd_add_test(scalar_traits_test utils SERIAL)

if(PLATFORM STREQUAL "CUDA")
    add_executable(scalar_traits_cuda_test.bin test_scalar_traits_cuda.cu)
    scfd_add_test(scalar_traits_cuda_test utils CUDA)
endif()

find_path(SCFD_BOOST_INCLUDE_DIR boost/multiprecision/cpp_bin_float.hpp
    HINTS "${Boost_ROOT}" "${BOOST_ROOT}" PATH_SUFFIXES include)
if(SCFD_BOOST_INCLUDE_DIR)
    add_executable(boost_scalar_traits_test.bin test_scalar_traits.cpp)
    target_compile_features(boost_scalar_traits_test.bin PRIVATE cxx_std_17)
    target_include_directories(boost_scalar_traits_test.bin PRIVATE ${SCFD_BOOST_INCLUDE_DIR})
    target_compile_definitions(boost_scalar_traits_test.bin PRIVATE SCFD_TEST_WITH_BOOST)
    scfd_add_test(boost_scalar_traits_test utils SERIAL)

    add_executable(boost_scalar_traits_isolation_test.bin test_scalar_traits.cpp)
    target_compile_features(boost_scalar_traits_isolation_test.bin PRIVATE cxx_std_17)
    target_include_directories(boost_scalar_traits_isolation_test.bin PRIVATE ${SCFD_BOOST_INCLUDE_DIR})
    target_compile_definitions(boost_scalar_traits_isolation_test.bin PRIVATE SCFD_TEST_BOOST_ISOLATION)
    scfd_add_test(boost_scalar_traits_isolation_test utils SERIAL)
endif()
