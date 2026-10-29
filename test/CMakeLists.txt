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


include(${PROJECT_SOURCE_DIR}/cmake/SCFDTesting.cmake)

add_subdirectory(utils)

add_subdirectory(external_libraries)

add_subdirectory(arrays)

add_subdirectory(static_vec)

add_subdirectory(static_mat)

add_subdirectory(for_each)

add_subdirectory(geometry)

add_subdirectory(for_paper)

add_subdirectory(matmul)

add_subdirectory(communication)
