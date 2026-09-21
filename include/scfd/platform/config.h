// Copyright © 2016-2026 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_PLATFORM_CONFIG_H__
#define __SCFD_PLATFORM_CONFIG_H__

#include <cstddef>

// Configuration for backend and platform headers only.
// Define ordinal macros before including SCFD headers to override the defaults.
// PLATFORM_ORDINAL controls local algorithms; PLATFORM_BIG_ORDINAL is the
// platform's global-index type and does not change MPI ranks or count handling.
// Define PLATFORM_MPI only for MPI platforms; leave it undefined otherwise.
#ifndef PLATFORM_ORDINAL
#    define PLATFORM_ORDINAL int
#endif

#ifndef PLATFORM_BIG_ORDINAL
#    define PLATFORM_BIG_ORDINAL std::ptrdiff_t
#endif

#endif
