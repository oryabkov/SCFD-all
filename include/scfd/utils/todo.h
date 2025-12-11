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

#ifndef __SCFD_UTILS_TODO_H__
#define __SCFD_UTILS_TODO_H__

#include <stdexcept>
#include <string>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define SCFD_TODO(X)                                                                                                                                                    \
    do {                                                                                                                                                           \
        throw std::runtime_error(std::string("TODO: " __FILE__ " " __STR(__LINE__) " : " #X " ")  );                                                        \
    } while (0)

#endif

#define SCFD_ATODO(X) assert(false && #X)


    